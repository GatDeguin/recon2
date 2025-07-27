import numpy as np
import torch
import torch.nn as nn
from .stgcn import STGCNBlock

class SHuBERT(nn.Module):
    """Simple self-supervised model that masks landmark and optical-flow streams."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, num_nodes: int = 544):
        super().__init__()
        A = np.eye(num_nodes, dtype=np.float32)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        self.layer1 = STGCNBlock(in_channels, hidden_dim, A)
        self.layer2 = STGCNBlock(hidden_dim, hidden_dim, A)
        self.layer3 = STGCNBlock(hidden_dim, hidden_dim, A)
        self.project = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        landmark_mask: torch.Tensor | None = None,
        flow_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional masks.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence ``(N, C, T, V)``.
        landmark_mask : torch.Tensor | None
            Boolean mask ``(N, T)`` for landmark nodes.
        flow_mask : torch.Tensor | None
            Boolean mask ``(N, T)`` for the optical-flow node.
        Returns
        -------
        torch.Tensor
            Reconstructed features with the same shape as ``x``.
        """
        n, c, t, v = x.shape
        if landmark_mask is not None:
            m = landmark_mask[:, None, :, None]
            x[:, :, :, :-1] = x[:, :, :, :-1] * (~m)
        if flow_mask is not None:
            m = flow_mask[:, None, :, None]
            x[:, :, :, -1:] = x[:, :, :, -1:] * (~m)
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        out = self.project(x)
        return out
