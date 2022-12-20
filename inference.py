import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from mobileone import reparameterize_model
import kornia
from torchvision import transforms as T
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import onnx
import onnxruntime as rt


class OverlayWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(OverlayWrapperModel, self).__init__()
        self.net = model
        
    def forward(self, input_stacked):
        input_img, overlay, alpha = torch.split(input_stacked, [3, 3, 1], dim=1)

        output_logits = self.net(torch.nn.functional.interpolate(input_img, (256, 512), mode="nearest")).sigmoid()
        
        
        alpha[torch.nn.functional.interpolate(output_logits, (1080, 1340), mode="nearest") > 0.5] = 0
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

        def shared_step(self, batch, stage):
            
            # if not self.global_step:
            #     self.global_step = 0
            # else:
            #     if stage == "train":
            #         self.global_step += 1
            
            true_fgr = batch["true_fgr"]
            true_pha = batch["true_pha"]
            true_bgr = batch["true_bgr"]


            true_src = true_bgr.clone()
            augment = False
            
            # Augment with shadow
            if augment and stage == "train":
                aug_shadow_idx = torch.rand(len(true_src)) < 0.3
                if aug_shadow_idx.any():
                    aug_shadow = true_pha[aug_shadow_idx].mul(0.3 * random.random())
                    aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=(-5, 5))(aug_shadow)
                    aug_shadow = kornia.filters.box_blur(aug_shadow, (random.choice(range(20, 40)),) * 2)
                    true_src[aug_shadow_idx] = true_src[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
                    del aug_shadow
                del aug_shadow_idx
            
            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)

            if augment and stage == "train":
            # Augment with noise
                aug_noise_idx = torch.rand(len(true_src)) < 0.4
                if aug_noise_idx.any():
                    true_src[aug_noise_idx] = true_src[aug_noise_idx].add_(torch.randn_like(true_src[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
                    true_bgr[aug_noise_idx] = true_bgr[aug_noise_idx].add_(torch.randn_like(true_bgr[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
                del aug_noise_idx
                
                # Augment background with jitter
                aug_jitter_idx = torch.rand(len(true_src)) < 0.8
                if aug_jitter_idx.any():
                    true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(0.18, 0.18, 0.18, 0.1)(true_bgr[aug_jitter_idx])
                del aug_jitter_idx
                
                # Augment background with affine
                aug_affine_idx = torch.rand(len(true_bgr)) < 0.3
                if aug_affine_idx.any():
                    true_bgr[aug_affine_idx] = T.RandomAffine(degrees=(0.0), translate=(0.01, 0.01))(true_bgr[aug_affine_idx])
                del aug_affine_idx
            


            # true_fgr, true_pha, true_bgr = random_crop(true_fgr, true_pha, true_bgr)
            # true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
            # true_pha[true_pha > 0] = 1
            # Shape of the image should be (batch_size, num_channels, height, width)
            # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
            assert true_src.ndim == 4

            # Check that image dimensions are divisible by 32, 
            # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
            # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
            # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
            # and we will get an error trying to concat these features
            h, w = true_src.shape[2:]
            assert h % 32 == 0 and w % 32 == 0



            # Shape of the mask should be [batch_size, num_classes, height, width]
            # for binary segmentation num_classes = 1
            assert true_pha.ndim == 4

            # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
            assert true_pha.max() <= 1.0 and true_pha.min() >= 0


            logits_mask = self.forward(true_src)
            
            # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
            loss = self.loss_fn(logits_mask, true_pha)

            self.log("loss", loss)
            

            if stage == "valid":
                imgs = []

                for img_idx in range(true_src.shape[0]):
                    imgs.extend([true_src[img_idx], true_pha[img_idx].expand(3, 256, 512), logits_mask[img_idx].sigmoid().expand(3, 256, 512)])
                grid = utils.make_grid(imgs, nrow=3)
                self.logger.experiment.add_image('masks', grid, self.global_step)

            # Lets compute metrics for some threshold
            # first convert mask values to probabilities, then 
            # apply thresholding
            prob_mask = logits_mask.sigmoid()
            pred_mask = (prob_mask > 0.5).float()

            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), true_pha.long(), mode="binary")

            return {
                "loss": loss,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }

        def shared_epoch_end(self, outputs, stage):
            # aggregate step metics
            tp = torch.cat([x["tp"] for x in outputs])
            fp = torch.cat([x["fp"] for x in outputs])
            fn = torch.cat([x["fn"] for x in outputs])
            tn = torch.cat([x["tn"] for x in outputs])

            # per image IoU means that we first calculate IoU score for each image 
            # and then compute mean over these scores
            per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
            
            # dataset IoU means that we aggregate intersection and union over whole dataset
            # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
            # in this particular case will not be much, however for dataset 
            # with "empty" images (images without target class) a large gap could be observed. 
            # Empty images influence a lot on per_image_iou and much less on dataset_iou.
            dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

            metrics = {
                f"{stage}_per_image_iou": per_image_iou,
                f"{stage}_dataset_iou": dataset_iou,
            }
            
            self.log_dict(metrics, prog_bar=True)

        def training_step(self, batch, batch_idx):
            return self.shared_step(batch, "train")            

        def training_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "train")

        def validation_step(self, batch, batch_idx):
            return self.shared_step(batch, "valid")

        def validation_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "valid")

        def test_step(self, batch, batch_idx):
            return self.shared_step(batch, "test")  

        def test_epoch_end(self, outputs):
            return self.shared_epoch_end(outputs, "test")

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.0002)



model = PetModel("deeplabv3plus", "mobileone_s0", in_channels=3, out_classes=1)
model = reparameterize_model(model)

checkpoint = torch.load(os.path.join(".", "whamen_imgs", "deeplab_dice_finetuned.pth"))
model.load(checkpoint=checkpoint)

wrapper_model = OverlayWrapperModel(model)

wrapper_model.eval()

wrapper_model = torch.quantization.quantize_dynamic(
    wrapper_model,  # the original model
    {torch.nn.Linear},  # a set of layers to dynamically quantize
    dtype=torch.qint8) 

transform =  transforms.ToTensor()

whamen_image = Image.open(os.path.join(".", "whamen_imgs", "image (5).png")).convert("RGB").resize((512, 256))

whamen_image = transform(whamen_image).unsqueeze(0)

import cv2
import numpy as np
import glob
import time
from openvino.runtime import Core
cap = cv2.VideoCapture("whamen_test.mp4")


overlay_imgs = sorted(glob.glob("C:\\Users\\kbond\\OneDrive\\Desktop\\projects\\streamfog\\res_old\\operator\\*.png"))

overlay_imgs = [transform(Image.open(img_name).convert("RGBA").resize((512, 256))) for img_name in overlay_imgs]

overlay_idx = 20


# sess_options = rt.SessionOptions()

# Set graph optimization level
# sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

# To enable model serialization after graph optimization set this
# sess_options.optimized_model_filepath = "wrapped_model.onnx"

# ort_session = rt.InferenceSession("wrapped_model.onnx", sess_options)


ort_session = rt.InferenceSession("wrapped_model.onnx")
# ort_session = rt.InferenceSession("wrapped_model.onnx", providers=['CUDAExecutionProvider'])


'''
model = onnx.load('regular_model.onnx')
output =[node.name for node in model.graph.output]

input_all = [node.name for node in model.graph.input]
input_initializer =  [node.name for node in model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)
'''



inference = "onnx"

device = torch.device("cuda")
# wrapper_model.to(device)


ie = Core()
model_ir = ie.read_model(model="wrapped_model\\wrapped_model.xml")
compiled_model = ie.compile_model(model=model_ir, device_name="CPU")

output_layer_ir = compiled_model.output(0)

with torch.no_grad():
    while(cap.isOpened()):
        success, img = cap.read()

        currrent_overlay = overlay_imgs[overlay_idx]

        overlay_idx += 1

        if overlay_idx == len(overlay_imgs):
            overlay_idx = 0

        img = img[:,360:1700,:]

        
        im_pil = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(im_pil).convert("RGB").resize((512, 256))

        stacked_input = torch.cat((transform(im_pil), currrent_overlay), dim=0)
       #  stacked_input.to(device)

        start = time.time()

       
        if inference == "torch":
            output = wrapper_model(stacked_input.unsqueeze(0))
        elif inference == "onnx":
            # input = stacked_input.unsqueeze(0).numpy()
            mid = time.time()
            print(f"preprocessing: {mid-start}")
            
            # output = ort_session.run(
            #     None,
            #     {'onnx::Sub_0': transform(input_img).unsqueeze(0).numpy()},
            # )
            output = ort_session.run(
                None,
                {'tensor': stacked_input.unsqueeze(0).numpy()},
            )
        elif inference == "mo":
            output = compiled_model(stacked_input.unsqueeze(0))[output_layer_ir]

        end = time.time()

        print((end-start))
        if inference == "torch":
            cv2.imshow("test",  (cv2.cvtColor(output.squeeze().numpy().transpose(1,2,0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))
        elif inference == "onnx":
            cv2.imshow("test",  (cv2.cvtColor(output[0].squeeze().transpose(1,2,0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))        
        elif inference == "mo":
            cv2.imshow("test",  (cv2.cvtColor(output.squeeze().transpose(1,2,0), cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))
        if cv2.waitKey(5) & 0xFF == 27:
            break
