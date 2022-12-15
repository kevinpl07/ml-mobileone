import os
import torch
import matplotlib.pyplot as plt
import kornia
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from pprint import pprint
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from torchvision.transforms.functional import center_crop, resize
from torchvision import transforms as T
from torchvision import utils
from segmentation_models_pytorch.datasets import VideoMattingDataset
from pytorch_lightning.loggers import TensorBoardLogger
from mobileone import mobileone, reparameterize_model
from PIL import Image
# download data
import random
import numpy as np
import json

class OverlayWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(OverlayWrapperModel, self).__init__()
        self.net = model
        
    def forward(self, input_stacked):
        input_img, overlay, alpha = torch.split(input_stacked, [3, 3, 1], dim=1)
        output_logits = self.net(input_img).sigmoid()
        alpha[output_logits > 0.5] = 0
        return  torch.nn.functional.interpolate(overlay * alpha + input_img * (1 - alpha), (720, 1280), mode="nearest")

    # def forward(self, input_img, overlay_img):
    #     overlay, alpha = torch.split(overlay_img, [3, 1], dim=1)
    #     output_logits = self.net(input_img).sigmoid()
    #     alpha[output_logits > 0.5] = 0
    #     return  torch.nn.functional.interpolate(overlay * alpha + input_img * (1 - alpha), (720, 1280), mode="bilinear", align_corners=True)



if __name__ == '__main__':

    root = "/Users/kev/Downloads/VideoMatte240K_JPEG_HD"
    backgrounds = "/Users/kev/Desktop/projects/kevs_playground/whamen_imgs/bgs/"

    # init train, val, test sets
    train_dataset = VideoMattingDataset(root, backgrounds, "train")
    valid_dataset = VideoMattingDataset(root, backgrounds, "valid")
    test_dataset = VideoMattingDataset(root, backgrounds, "test")

    # # It is a good practice to check datasets don`t intersects with each other
    assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
    assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
    assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

    print(f"Train size: {len(train_dataset)}")
    print(f"Valid size: {len(valid_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    n_cpu = os.cpu_count() // 2
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=n_cpu)

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
            self.model = reparameterize_model(model)

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

    tb_logger = TensorBoardLogger(save_dir="logs/")
    trainer = pl.Trainer(
        logger=tb_logger,
        log_every_n_steps=1,
        val_check_interval=100,
        limit_val_batches=1,
        max_time="00:05:00:00"
    )

    model = PetModel("deeplabv3plus", "mobileone_s0", in_channels=3, out_classes=1)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )

    torch.save(model.state_dict(), "deeplab_dice.pth")
    
    # checkpoint = torch.load("model_m1_s0_pan.pth")
    # model.load(checkpoint=checkpoint)
    # model.load_state_dict(checkpoint, strict=False)
    model.eval()
    model = reparameterize_model(model)

    # wrapper_model = OverlayWrapperModel(model)

    # dummy_input = torch.rand((1, 7, 256, 512))
    # torch.onnx.export(wrapper_model, dummy_input, "wrapped_model.onnx", opset_version=11)
    # import pdb; pdb.set_trace()
    # valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
    # print(valid_metrics)

    # # run test dataset
    # test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
    # print(test_metrics)

    # bgr_image = Image.open("./background.png").convert("RGBA").resize((512, 256))
    # bgr_image.save("background_resized.png")
    transform =  transforms.ToTensor()

    whamen_image = Image.open("/Users/kev/Desktop/projects/kevs_playground/whamen_imgs/image (5).png").convert("RGB").resize((512, 256))

    whamen_image = transform(whamen_image).unsqueeze(0)

    # batch = next(iter(test_dataloader))
    import time

    # test_img = "/Users/kev/Desktop/projects/kevs_playground/whamen_imgs/bgs/"

    # true_fgrs = batch["true_fgr"]
    # true_phas = batch["true_pha"]
    # true_bgrs = batch["true_bgr"]
    # true_bgr_phas = batch["true_bgr_pha"]
   
    
    # true_fgr, true_pha, true_bgr = random_crop(true_fgr, true_pha, true_bgr)
    # true_srcs = true_fgrs * true_phas + true_bgrs * (1 - true_phas)

    stacked = False
    with torch.no_grad():
        if stacked:
        
            stacked_input = torch.cat((true_srcs, bgr_image.expand(16, 4, 256, 512)), dim=1)
            true_phas[true_phas > 0] = 1
            start = time.time()
            logits = wrapper_model(stacked_input)
            end = time.time()
            print(end-start)
        else:
            logits = model(whamen_image).sigmoid()

    #pr_masks = logits.sigmoid()
    print(logits.shape)
    show = True
    idx = 0
    

    if show:
        plt.figure(figsize=(10, 5))
        # true_fgr = true_fgr.unsqueeze(0)
        # true_pha = true_pha.unsqueeze(0)
        # true_bgr = true_bgr.unsqueeze(0)
        
        # true_src = true_fgr * true_pha + true_bgr * (1 - true_pha)
        # true_pha[true_pha > 0] = 1

        plt.subplot(1, 3, 1)
        plt.imshow(whamen_image.numpy().squeeze().transpose(1, 2, 0))  # convert CHW -> HWC
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(logits.numpy().squeeze()) # just squeeze classes dim, because we have only one class
        plt.title("Ground truth")
        plt.axis("off")

        # plt.subplot(1, 3, 3)
        # plt.imshow(pr_mask.numpy().transpose(1, 2, 0)) # just squeeze classes dim, because we have only one class
        # plt.title("Ground truth")
        # plt.axis("off")

        plt.show()
    else:
        img = pr_mask.numpy().squeeze().transpose(1, 2, 0)
        print(img.shape)
        im = Image.fromarray((img*255).astype(np.uint8))
        im.save(f"output/{idx}.png")
        idx += 1

