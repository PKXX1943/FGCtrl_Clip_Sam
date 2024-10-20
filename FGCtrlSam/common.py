import torch
import torch.nn as nn
from segment_anything.modeling.common import LayerNorm2d, MLPBlock

class LayerNorm1d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(2, keepdim=True)  
        s = (x - u).pow(2).mean(2, keepdim=True)  
        x = (x - u) / torch.sqrt(s + self.eps)  
        x = self.weight[:, None] * x + self.bias[:, None]  
        return x
