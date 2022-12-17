import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import os
from mobileone import reparameterize_model

class OverlayWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(OverlayWrapperModel, self).__init__()
        self.net = model
        
    def forward_upsampling(self, input_stacked):
        input_img, overlay, alpha = torch.split(input_stacked, [3, 3, 1], dim=1)

        output_logits = self.net(torch.nn.functional.interpolate(input_img, (256, 512), mode="nearest")).sigmoid()
        
        alpha[torch.nn.functional.interpolate(output_logits, (1080, 1340), mode="bilinear") > 0.5] = 0
        return overlay * alpha + input_img * (1 - alpha)

    def forward(self, input_stacked):
        input_img, overlay, alpha = torch.split(input_stacked, [3, 3, 1], dim=1)
        output_logits = self.net(input_img).sigmoid()
        alpha[output_logits > 0.5] = 0
        return overlay * alpha + input_img * (1 - alpha)



class PetModel(pl.LightningModule):

        def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
            super().__init__()
            self.model = smp.create_model(
                arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
            )
            
            # preprocessing parameteres for image
            params = smp.encoders.get_preprocessing_params(encoder_name)
            self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
            self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

            # for image segmentation dice loss could be the best first choice
            self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
            # self.loss_fn = smp.losses.FocalLoss(smp.losses.BINARY_MODE, gamma=0)

        def load(self, checkpoint):
            import collections
            new_state_dict = collections.OrderedDict()
            for k, v in checkpoint.items():
                name = k.replace("model.", '') # remove `module.`
                new_state_dict[name] = v
            new_state_dict.pop("std")
            new_state_dict.pop("mean")
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            # self.model = reparameterize_model(model)

        def forward(self, image):
            # normalize image here
            image = (image - self.mean) / self.std
            mask = self.model(image)
            return mask

        


model = PetModel("deeplabv3plus", "mobileone_s0", in_channels=3, out_classes=1)
model.eval()
model = reparameterize_model(model)


checkpoint = torch.load(os.path.join(".", "whamen_imgs", "deeplab_dice_finetuned.pth"))
model.load(checkpoint=checkpoint)
model.eval()

wrapper_model = OverlayWrapperModel(model)
wrapper_model.eval()


torch.onnx.export(wrapper_model, torch.rand(1, 7, 256, 512), "wrapped_model.onnx", opset_version=11)

torch.onnx.export(model, torch.rand(1, 3, 256, 512), "regular_model.onnx", opset_version=11)