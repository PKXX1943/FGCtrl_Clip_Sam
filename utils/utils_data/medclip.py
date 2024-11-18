from model.clip_encoder import ClipEncoder
from PIL import Image   
from sklearn.manifold import TSNE
import torch
import math
import torchshow as ts
import os
import json
import numpy as np

class ClipInference(ClipEncoder):
    def forward(
        self, 
        batch_images: list,
        batch_captions: list,
        n_patches: int = 2,
    ):
        assert len(batch_captions) == len(batch_images)
        bs = len(batch_images)
        caption_length = len(batch_captions[0].split('\n'))
        images = []
        captions = []
        for image, caption in zip(batch_images, batch_captions):
            target_size = max(image.size)
            images.extend(self.get_patches(self.resize_image(image, target_size), n_patches))
            # images.extend(self.get_masked(self.resize_image(image, target_size), n_patches))
            captions.extend(caption.strip().split('\n'))
            images.append(self.resize_image(image, target_size))
        image_tensor = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        text_tensor = self.tokenizer(captions, context_length=self.context_length).to(self.device)
        with torch.no_grad():
            self.model.visual.trunk.norm.register_forward_hook(self.hook_fn)
            pooled = self.model.encode_image(image_tensor, normalize=True)
            text_embedding = self.model.encode_text(text_tensor, normalize=True)
        pooled = pooled.view(bs, n_patches*n_patches+1, -1)
        image_embedding = self.interm_embeddings.view(bs, n_patches*n_patches+1, self.interm_embeddings.shape[-2], self.interm_embeddings.shape[-1])
        text_embedding = text_embedding.view(bs, caption_length, -1)
        # Calculate cosine similarity
        similarities = torch.zeros(bs, caption_length).to(self.device)
        for i in range(bs):
            for j in range(caption_length):
                text_emb = text_embedding[i, j]
                cos_sim = torch.nn.functional.cosine_similarity(pooled[i], text_emb.unsqueeze(0), dim=-1)
                similarities[i, j] = cos_sim.max()

        return image_embedding, similarities

def visualize_tsne(features: np.ndarray, n_components=2) -> Image:
    features_reshaped = features.reshape(-1, 768)
    tsne = TSNE(n_components=n_components, random_state=42)
    features_tsne = tsne.fit_transform(features_reshaped)
    features_tsne = features_tsne.reshape(14, 14, n_components)
    if n_components == 2:
        features_tsne = (features_tsne - features_tsne.min()) / (features_tsne.max() - features_tsne.min()) * 255
        features_tsne = features_tsne.astype(np.uint8)  
        image = Image.fromarray(features_tsne[..., 0])  
    elif n_components == 3:
        features_tsne = (features_tsne - features_tsne.min()) / (features_tsne.max() - features_tsne.min()) * 255
        features_tsne = features_tsne.astype(np.uint8)  
        image = Image.fromarray(features_tsne) 
    
    return image

    
annotation_file = 'data/annotations/tuning.json'
with open(annotation_file, 'r') as f:   
    data = json.load(f)
images = []
captions = []
for item in data:
    img = Image.open(item['image_path'])
    caption = '\n'.join([val for val in item['analysis_result'].values()])
    images.append(img)
    captions.append(caption)
clip = ClipInference().to('cuda')
num_test=10
image_embedding, similarities = clip(images[:num_test], captions[:num_test])
grid = int(math.sqrt(image_embedding.shape[2]))
print(similarities)
image_embedding = image_embedding[:,-1,:-1,:].view(image_embedding.shape[0], grid, grid, -1)[:,:,:,:3]
for ii in range(image_embedding.shape[0]):
    vis = visualize_tsne(image_embedding[ii].cpu().numpy())
    vis.save(f'debug/clip_embedding_{ii:02d}.png')

