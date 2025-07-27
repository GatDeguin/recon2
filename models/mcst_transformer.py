import torch
import torch.nn as nn


class MCSTBlock(nn.Module):
    """Spatio-temporal block with multi-scale temporal attention."""

    def __init__(self, dim: int, num_heads: int = 4, kernels: tuple[int, ...] = (3, 5, 7)):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spatial_ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.temporal_ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )
        self.q_convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k // 2) for k in kernels])
        self.k_convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k // 2) for k in kernels])
        self.v_convs = nn.ModuleList([nn.Conv1d(dim, dim, k, padding=k // 2) for k in kernels])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        # Spatial attention
        s = x.permute(0, 2, 3, 1).reshape(n * t, v, c)
        attn_s, _ = self.spatial_attn(s, s, s)
        s = s + attn_s
        s = s + self.spatial_ff(s)
        s = s.reshape(n, t, v, c).permute(0, 3, 1, 2)
        # Temporal multi-scale attention
        tmp = s.permute(0, 3, 2, 1).reshape(n * v, t, c)
        base = tmp
        tmp_feat = tmp.transpose(1, 2)  # (N*V, C, T)
        outs = []
        for q_conv, k_conv, v_conv in zip(self.q_convs, self.k_convs, self.v_convs):
            q = q_conv(tmp_feat).transpose(1, 2)
            k = k_conv(tmp_feat).transpose(1, 2)
            v_ = v_conv(tmp_feat).transpose(1, 2)
            o, _ = self.temporal_attn(q, k, v_)
            outs.append(o)
        attn_t = sum(outs) / len(outs)
        tmp = base + attn_t
        tmp = tmp + self.temporal_ff(tmp)
        out = tmp.reshape(n, v, t, c).permute(0, 3, 2, 1)
        return self.relu(out)


class MCSTTransformer(nn.Module):
    """Multi-Scale Channel Spatio-Temporal Transformer."""

    def __init__(self, in_channels: int, num_class: int, num_nodes: int, num_layers: int = 2, embed_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.layers = nn.ModuleList([MCSTBlock(embed_dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)  # (N, C, T, 1)
        feat = x.squeeze(-1).permute(0, 2, 1)
        out = self.fc(feat)
        log_probs = out.log_softmax(-1)
        if return_features:
            return log_probs, feat
        return log_probs
