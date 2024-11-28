import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple


from functools import partial

from .clip_encoder import ClipEncoder
from .decoder import ClipCellDecoder
from segment_anything.modeling import ImageEncoderViT

class ClipCellSam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"
    
    def __init__(
        self,
        sam_image_encoder: ImageEncoderViT,
        clip_encoder: ClipEncoder,
        ClipCell_decoder: ClipCellDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.image_encoder = sam_image_encoder
        self.clip_encoder = clip_encoder
        self.decoder = ClipCell_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        
    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(
        self,
        batched_input : dict,
        lamda: float = 0.5,
        n_patches: int = None,
        multimask_output: bool = False
    ):
        # image_inputs = self.preprocess(batched_input["image"].to(self.device))
        image_inputs = batched_input["image"].to(self.device)
        with torch.no_grad():
            image_embedding, interm_embeddings = self.image_encoder(image_inputs)
        pil_images = batched_input["pil_image"]
        captions = batched_input["caption"]
        clip_embedding, img_token, pos_token, neg_token = self.clip_encoder(
            pil_images, captions, n_patches
        )
        mask_logits, iou_predictions = self.decoder(
            image_embedding, interm_embeddings, clip_embedding, img_token, pos_token,  multimask_output
        )
        mask_bg, _ = self.decoder(
            image_embedding, interm_embeddings, clip_embedding, img_token, neg_token,  multimask_output
        )
        masks = self.postprocess_masks(
                mask_logits,
                mask_bg,
                lamda,
                original_size=batched_input["image"].shape[-2:],
            )
        masks = masks > self.mask_threshold

        output = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "logits": mask_logits,
            "bg": mask_bg
        }
        
        return output
    
    # copied from segment_anything/modeling/sam.py
    def postprocess_masks(
        self,
        mask_logits: torch.Tensor,
        mask_bg: torch.Tensor,
        lamda: float,
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
        # masks = F.interpolate(
        #     masks,
        #     (self.image_encoder.img_size, self.image_encoder.img_size),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        # masks = masks[..., : input_size[0], : input_size[1]]
        # mean = masks.mean(dim=(2, 3), keepdim=True)  
        # masks = masks - mean
        logits = lamda * mask_bg + (lamda - 1) * mask_logits
        masks_ori = F.interpolate(logits, original_size, mode="bilinear", align_corners=False)
        # masks = F.relu(masks)
        # masks_ori = F.relu(masks_ori)
        return masks_ori

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