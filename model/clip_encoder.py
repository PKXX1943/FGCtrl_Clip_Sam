import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Type, Tuple, Optional, List
from PIL import Image
import math
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer
from open_clip.model import VisionTransformer, TimmModel

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
        
    @property
    def device(self):
        return next(self.parameters()).device

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
            torch.arange(n_patches, dtype=torch.float32, device=self.device),  # rows
            torch.arange(n_patches, dtype=torch.float32, device=self.device)   # columns
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
        transformer_dim: int,
        vit_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.pos_embed = PositionEmbeddingRandom(vit_dim//2,)
        self.lin1_img = nn.Linear(embedding_dim, vit_dim)
        self.lin1_img_pooled = nn.Linear(embedding_dim, transformer_dim)
        self.lin1_text = nn.Linear(embedding_dim, vit_dim)
        # self.norm_pooled =nn.LayerNorm(out_dim)
        self.norm_img =nn.LayerNorm(vit_dim)
        # self.norm_text =nn.LayerNorm(out_dim)
        self.act = act()
        
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
        pooled: torch.Tensor,
        text_embedding: torch.Tensor
        ):
        pe = self.pos_embed(n_patches)
        img_out = self.lin1_img_pooled(pooled)
        text_out = self.lin1_text(text_embedding)
        img_embedding = self.norm_img(self.lin1_img(image_embedding))
        
        return img_embedding, img_out, text_out, pe.unsqueeze(0).expand(img_out.size(0), -1, -1)

class ClipEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        clip_model: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        tokenizer: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        context_length : int = 256,
        learnable_pe: bool = True,
    ):
        """
        Biomed-clip inference model to generate clip features for image patches and caption. Using linear head 
        to align dimensions of clip features with the ImageEncoder output.

        Args:
          embedding_dim : the channel dimension of the clip features
          out_dim : the channel dimension of the ImageEncoder output features
          clip_model : the huggingface repo of the biomed-clip
          tokenizer : the tokenizer used by biomed-clip
          learnable_pe : whether to use a learnable position embedding for patch features
          num_text_embedding : number of text embeddings from clip model
        """
        super().__init__()
        
        self.model, self.preprocess = create_model_from_pretrained(clip_model)
        # self.model.visual = ClipTimm_interm(**self.model.visual.__dict__) 
        self.tokenizer = get_tokenizer(tokenizer)
        self.context_length = context_length
        self.interm_embeddings = None
        self.model.eval()
        
    @property
    def device(self):
        return next(self.parameters()).device
    
    def hook_fn(self, moudle, input, output):
        self.interm_embeddings = output

    def auto_patch(self, size):
        resolution, _ = size
        if resolution < 256:
            return 0
        elif resolution < 512:
            return 1
        else:
            return 2

    def get_patches(self, image : Image, n_patches : int):  
        # split an image into patches   
        width, height = image.size
        block_size = width // n_patches
        blocks = []
        for i in range(0, width, block_size):
            for j in range(0, height, block_size):
                box = (j, i, j + block_size, i + block_size)  
                block = image.crop(box)  
                blocks.append(block)
        return blocks
    
    def get_masked(self, image: Image, n_patches: int):
        # mask different region of an image
        width, height = image.size
        block_size = width // n_patches
        image_np = np.array(image)
        blocks = []
        for center_i in range(n_patches):
            for center_j in range(n_patches):
                modified_image_np = np.zeros_like(image_np)            
                for i in range(n_patches):
                    for j in range(n_patches):
                        top_left_x = j * block_size
                        top_left_y = i * block_size
                        bottom_right_x = (j + 1) * block_size
                        bottom_right_y = (i + 1) * block_size
                        if i == center_i and j == center_j:
                            modified_image_np[
                                top_left_y:bottom_right_y, top_left_x:bottom_right_x] = \
                                    image_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                        elif (i > 0 and i < n_patches-1) and (j > 0 and j < n_patches-1):
                            modified_image_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = \
                                np.clip(image_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x] * 0.5, 0, 255)
                        else:
                            modified_image_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 0
                blocks.append(Image.fromarray(modified_image_np.astype(np.uint8)))
        return blocks
    
    def resize_image(self, image, target_size=1024):
        # resize method that is consistent with dataset transforms
        transform_to_tensor = T.ToTensor()  
        image_tensor = transform_to_tensor(image).unsqueeze(0)  
        
        resized_tensor = F.interpolate(image_tensor, size=(target_size, target_size), mode='bilinear')
        
        transform_to_pil = T.ToPILImage()  
        resized_image = transform_to_pil(resized_tensor.squeeze(0))  
        
        return resized_image
    
    def cal_similarities(self, image_features, text_feature):
        return F.cosine_similarity(image_features, text_feature.unsqueeze(0), dim=1)   
    
    def forward(
        self, 
        batch_images: list,
        batch_captions: list,
        n_patches: int = None,
        ):
        assert len(batch_captions) == len(batch_images)
        bs = len(batch_images)
        caption_length = len(batch_captions[0].split('\n'))
        images = []
        captions = []
        if n_patches is None:
            n_patches = self.auto_patch(image.size)
        assert n_patches in [2**k for k in range(0, 5)], f"n_patches must eqaul 2 ^ k while k=0,1,2,3,4."
        for image, caption in zip(batch_images, batch_captions):
            target_size = max(image.size)
            images.extend(self.get_patches(self.resize_image(Image.fromarray(image), target_size), n_patches))
            # images.extend(self.get_masked(self.resize_image(image, target_size), n_patches))
            captions.extend(caption.strip().split('\n'))
            images.append(self.resize_image(image, target_size))
        image_tensor = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tensor = self.tokenizer(captions, context_length=self.context_length).to(self.device)
        with torch.no_grad():
            self.model.visual.trunk.norm.register_forward_hook(self.hook_fn)
            pooled = self.model.encode_image(image_tensor, normalize=True)
            # pooled = self.model.visual.trunk(image_tensor)
            # pooled = self.model.visual.head(pooled)
            text_embedding = self.model.encode_text(text_tensor, normalize=True)
        img_out = pooled.view(bs, n_patches*n_patches+1, -1)[:, -1, :]
        image_embedding = self.interm_embeddings.view(bs, n_patches*n_patches+1, self.interm_embeddings.shape[-2], self.interm_embeddings.shape[-1])
        text_out = text_embedding.view(bs, caption_length, -1)
        pos_text = text_out[:, :-1, :]
        neg_text = text_out[:, -1, :]
        return image_embedding, img_out, pos_text, neg_text

        
        
