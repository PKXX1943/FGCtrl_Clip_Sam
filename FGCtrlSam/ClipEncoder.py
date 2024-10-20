import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Type, Tuple, Optional, List
from PIL import Image
import math
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer

# lightly adapte from segment_anything/modeling/prompt_encoder.py
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies for n*n patch.
    """
    def __init__(self, num_pos_feats: int, learnable_pe: bool = True, scale: Optional[float] = None) -> None:
        super().__init__()
        self.embedding_dim = 2*num_pos_feats
        if scale is None or scale <= 0.0:
            self.scale = 1.0
        else:
            self.scale = scale
        self.positional_encoding_matrix = nn.Parameter(torch.randn(2, num_pos_feats), requires_grad=learnable_pe)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 and have shape d_1 x ... x d_n x 2
        coords = 2 * coords - 1  # scale to [-1, 1]
        coords = coords @ (self.scale * self.positional_encoding_matrix)  # apply random matrix
        coords = 2 * np.pi * coords  # scale
        # outputs d_1 x ... x d_n x C shape (C = 2 * num_pos_feats)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, n_patches: int) -> torch.Tensor:
        """Generate positional encoding for an n x n grid of patches."""
        # Create normalized coordinates for each patch in an n x n grid
        coords = torch.stack(torch.meshgrid(
            torch.arange(n_patches, dtype=torch.float32),  # rows
            torch.arange(n_patches, dtype=torch.float32)   # columns
        ), dim=-1)  # shape: n x n x 2 (i, j)

        # Normalize coordinates to [0,1] by dividing by n (patch coordinates are (i/n, j/n))
        coords = coords / n_patches

        # Apply positional encoding
        pe = self._pe_encoding(coords)  # shape: n x n x (2 * num_pos_feats)

        return pe.view(-1, self.embedding_dim)


class ClipHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        out_dim: int,
        learnable_pe: bool = True,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.pos_embed = PositionEmbeddingRandom(embedding_dim//2,)
        self.lin1_img = nn.Linear(embedding_dim, mlp_dim)
        self.lin2_img = nn.Linear(mlp_dim, out_dim)
        self.lin1_text = nn.Linear(embedding_dim, mlp_dim)
        self.lin2_text = nn.Linear(mlp_dim, out_dim)
        self.norm =nn.LayerNorm(out_dim)
        self.act = act()
        self.learnable_pe = learnable_pe
        
        # pos embeddings based on patch position
        # pos_embed = torch.zeros((n_patches*n_patches, embedding_dim))
        # for i in range(n_patches):
        #     for j in range(n_patches):
        #         pos = i * n_patches + j  # flatten
        #         for k in range(embedding_dim // 2):
        #             pos_embed[pos, 2 * k] = math.sin((i + j) / 10000 ** (2 * k / embedding_dim))
        #             pos_embed[pos, 2 * k + 1] = math.cos((i + j) / 10000 ** (2 * k / embedding_dim))
        # self.pos_embed = pos_embed
        
        # if learnable_pos:
        #     self.learnable_pos_embed = nn.Parameter(torch.zeros_like(n_patches*n_patches, embedding_dim)
            
    def forward(
        self,
        n_patches: int,
        image_embedding: torch.Tensor,
        text_embedding: torch.Tensor
        ):
        bs = image_embedding.size(0)
        pe = self.pos_embed(n_patches)
        img_out = self.norm(self.lin2_img(self.act(self.lin1_img(image_embedding))))
        text_out = self.norm(self.lin2_text(self.act(self.lin1_text(text_embedding))))
        
        return img_out, text_out, pe.expand(img_out.shape)
        

class ClipInference:
    def __init__(
        self,
        embedding_dim: int = 512,
        out_dim: int = 256,
        clip_model: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        tokenizer: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        device: str = None,
        learnable_pe: bool = True
    ):
        """
        Biomed-clip inference model to generate clip features for image patches and caption. Using linear head 
        to align dimensions of clip features with the ImageEncoder output.

        Args:
          embedding_dim : the channel dimension of the clip features
          out_dim : the channel dimension of the ImageEncoder output features
          clip_model : the huggingface repo of the biomed-clip
          tokenizer : the tokenizer used by biomed-clip
          device : the device to put the clip model and data on
          learnable_pe : whether to use a learnable position embedding for patch features
        """
        if device is not None:
            self.device = device       
        else:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            
        self.model, self.preprocess = create_model_from_pretrained(clip_model)
        self.tokenizer = get_tokenizer(tokenizer)
        self.head = ClipHead(
            embedding_dim=embedding_dim,
            mlp_dim=out_dim,
            out_dim=out_dim,
            learnable_pe=learnable_pe,
        ).to(self.device)
        self.model.to(self.device)
        self.model.eval()
        
        
    def get_patches(self, image:Image, n_patches):  
        # split an image into patches   
        assert n_patches in [2**k for k in range(1, 7)], f"n_patches must eqaul 2 ^ k while k=1,2,3,4,5,6."
        width, height = image.size
        block_size = width // n_patches
        blocks = []
        for i in range(0, width, block_size):
            for j in range(0, height, block_size):
                box = (j, i, j + block_size, i + block_size)  
                block = image.crop(box)  
                blocks.append(block)
        return blocks
    
    def resize_image(image, target_size=(1024, 1024)):
        # resize method that is consistent with dataset transforms
        transform_to_tensor = T.ToTensor()  
        image_tensor = transform_to_tensor(image).unsqueeze(0)  
        
        resized_tensor = F.interpolate(image_tensor, size=target_size, mode='bilinear')
        
        transform_to_pil = T.ToPILImage()  
        resized_image = transform_to_pil(resized_tensor.squeeze(0))  
        
        return resized_image

    
    def __call__(
        self, 
        batch_images: list,
        batch_captions: list,
        n_patches: int = 4,
        target_size: Tuple = (1024, 1024),
        context_length: int = 256,
        num_text_embedding: int = 1
        ):
        assert len(batch_captions) == len(batch_images)
        bs = len(batch_images)
        images = []
        captions = []
        for image, caption in zip(batch_images, batch_captions):
            images.extend(self.get_patches(self.resize_image(image, target_size), n_patches))
            captions.extend([caption] * (n_patches*n_patches + 1))
            images.append(self.resize_image(image, target_size))
        image_tensor = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tensor = self.tokenizer(captions, context_length=context_length).to(self.device)
        with torch.no_grad():
            image_embedding, text_embedding, _ = self.model(image_tensor, text_tensor)
        image_embedding.view(bs, n_patches*n_patches+1, -1)
        text_embedding.view(bs, n_patches*n_patches+1, -1)
        img_out, text_out, pe = self.head(n_patches, image_embedding, text_embedding[:, :num_text_embedding, :])
        return img_out, text_out, pe

        
        
