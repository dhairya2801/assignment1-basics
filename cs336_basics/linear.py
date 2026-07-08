import torch
import torch.nn as nn
import numpy as np
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        variance = 2 / (in_features + out_features)
        std = np.sqrt(variance)
        size = (out_features, in_features)

        # initialize weight matrix
        w = nn.Parameter(torch.empty(size, device=device, dtype=dtype))
        self.W = torch.nn.init.trunc_normal_(w, mean=0, std=std, a=-3, b=3, generator=None)
    
    def forward(self, x):
        # implement y = Wx; W: (dout, din), x: (din, 1)
        # in row major; y = xW_T

        y = einsum(x, self.W, "... din, dout din -> ... dout")
        return y


if __name__ == '__main__':
    obj = Linear(in_features=3, out_features=3)
    x = torch.rand(1,3)
    out = obj.forward(x)
    print(out)