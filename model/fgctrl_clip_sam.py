import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple


from functools import partial

from .clip_encoder import ClipEncoder
from .fgctrl_decoder import FGCtrlDecoder
from segment_anything.modeling import ImageEncoderViT

class FGCtrlClipSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        sam_image_encoder: ImageEncoderViT,
        clip_encoder: ClipEncoder,
        fgctrl_decoder: FGCtrlDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.image_encoder = sam_image_encoder
        self.clip_encoder = clip_encoder
        self.decoder = fgctrl_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(
        self,
        batched_input : dict,
        multimask_output: bool = False
    ):
        image_inputs = self.preprocess(batched_input["image"].to(self.device))
        with torch.no_grad():
            image_embedding = self.image_encoder(image_inputs)
        pil_images = batched_input["pil_image"]
        captions = batched_input["caption"]
        n_patches = self.decoder.n_patches
        clip_embedding, text_embedding, clip_pe = self.clip_encoder(
            pil_images, captions, n_patches
        )
        mask_logits, iou_predictions = self.decoder(
            image_embedding, clip_embedding, clip_pe, text_embedding, multimask_output
        )
        masks = self.postprocess_masks(
                mask_logits,
                input_size=mask_logits.shape[-2:],
                original_size=batched_input["image"].shape[-2:],
            )
        masks = masks > self.mask_threshold

        output = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "logits": mask_logits
        }
        
        return output
    
    # copied from segment_anything/modeling/sam.py
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
