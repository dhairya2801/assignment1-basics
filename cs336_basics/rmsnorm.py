import torch
import torch.nn as nn
import numpy as np
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float=1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        x_sum = reduce(x**2, 'batch seqlen d_model -> batch seqlen 1', 'sum')
        RMS_x = torch.sqrt((1/self.d_model)*(x_sum) + self.eps)
        RMS_norm = (x/RMS_x) * self.gain
        
        return RMS_norm.to(in_dtype)