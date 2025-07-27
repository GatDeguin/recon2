import torch
import torch.nn as nn

class STTNBlock(nn.Module):
    """Reduced Spatio-Temporal Transformer block."""
    def __init__(self, dim: int, num_heads: int = 4):
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
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T, V)
        n, c, t, v = x.shape
        # Spatial attention: attend over joints for each frame
        s = x.permute(0, 2, 3, 1).reshape(n * t, v, c)  # (N*T, V, C)
        attn_s, _ = self.spatial_attn(s, s, s)
        s = s + attn_s
        s = s + self.spatial_ff(s)
        s = s.reshape(n, t, v, c).permute(0, 3, 1, 2)  # (N, C, T, V)
        # Temporal attention: attend over frames for each joint
        tmp = s.permute(0, 3, 2, 1).reshape(n * v, t, c)  # (N*V, T, C)
        attn_t, _ = self.temporal_attn(tmp, tmp, tmp)
        tmp = tmp + attn_t
        tmp = tmp + self.temporal_ff(tmp)
        out = tmp.reshape(n, v, t, c).permute(0, 3, 2, 1)  # (N, C, T, V)
        return self.relu(out)


class STTN(nn.Module):
    """Simplified Spatio-Temporal Transformer Network with multitask heads."""
    def __init__(
        self,
        in_channels: int,
        num_class: int,
        num_nodes: int,
        num_layers: int = 2,
        embed_dim: int = 128,
        num_nmm: int = 0,
        num_suffix: int = 0,
        num_rnm: int = 0,
        num_person: int = 0,
        num_number: int = 0,
        num_tense: int = 0,
        num_aspect: int = 0,
        num_mode: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)
        self.layers = nn.ModuleList([STTNBlock(embed_dim) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.ctc_head = nn.Linear(embed_dim, num_class)
        self.nmm_head = nn.Linear(embed_dim, num_nmm) if num_nmm > 0 else None
        self.suffix_head = nn.Linear(embed_dim, num_suffix) if num_suffix > 0 else None
        self.rnm_head = nn.Linear(embed_dim, num_rnm) if num_rnm > 0 else None
        self.person_head = nn.Linear(embed_dim, num_person) if num_person > 0 else None
        self.number_head = nn.Linear(embed_dim, num_number) if num_number > 0 else None
        self.tense_head = nn.Linear(embed_dim, num_tense) if num_tense > 0 else None
        self.aspect_head = nn.Linear(embed_dim, num_aspect) if num_aspect > 0 else None
        self.mode_head = nn.Linear(embed_dim, num_mode) if num_mode > 0 else None

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        # x: (N, C, T, V)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)  # (N, C, T, 1)
        feat = x.squeeze(-1).permute(0, 2, 1)  # (N, T, C)
        gloss = self.ctc_head(feat).log_softmax(-1)
        pooled = feat.mean(dim=1)
        nmm = self.nmm_head(pooled) if self.nmm_head else None
        suffix = self.suffix_head(pooled) if self.suffix_head else None
        rnm = self.rnm_head(pooled) if self.rnm_head else None
        person = self.person_head(pooled) if self.person_head else None
        number = self.number_head(pooled) if self.number_head else None
        tense = self.tense_head(pooled) if self.tense_head else None
        aspect = self.aspect_head(pooled) if self.aspect_head else None
        mode = self.mode_head(pooled) if self.mode_head else None
        outputs = (gloss, nmm, suffix, rnm, person, number, tense, aspect, mode)
        if return_features:
            return outputs, feat
        return outputs
