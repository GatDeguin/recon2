import torch
import torch.nn as nn
from .stgcn import STGCN

class CorrNet(nn.Module):
    """Compute correlation maps between frames."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.scale = 1.0 / (in_channels ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        feat = x.view(n, c * v, t)
        feat = feat - feat.mean(dim=2, keepdim=True)
        corr = torch.bmm(feat.transpose(1, 2), feat) * self.scale
        return corr / (c * v)


class CorrNetPlus(nn.Module):
    """ST-GCN encoder enhanced with temporal correlation attention."""
    def __init__(self, in_channels: int, num_class: int, num_nodes: int):
        super().__init__()
        self.corr = CorrNet(in_channels)
        self.encoder = STGCN(in_channels, num_class, num_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        corr = self.corr(x)
        weights = torch.softmax(corr, dim=-1)  # (N, T, T)
        agg = torch.einsum('nts,ncsv->nctv', weights, x)
        x = x + agg
        return self.encoder(x)
