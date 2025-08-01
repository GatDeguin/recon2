import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from utils.build_adjacency import build_adjacency

class GraphConv(nn.Module):
    """Simple graph convolution using a fixed adjacency matrix."""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        x = torch.einsum('nctv,vw->nctw', x, self.A)
        return self.conv(x)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.gcn = GraphConv(in_channels, out_channels, A)
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(9,1), padding=(4,0)),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.down = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.down = lambda x: x

    def forward(self, x):
        res = self.down(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_class,
        num_nodes,
        num_nmm: int = 0,
        num_suffix: int = 0,
        num_rnm: int = 0,
        num_person: int = 0,
        num_number: int = 0,
        num_tense: int = 0,
        num_aspect: int = 0,
        num_mode: int = 0,
    ):
        """Spatial Temporal GCN with optional multitask heads."""
        super().__init__()
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "skeleton.yaml"
        if cfg_path.exists():
            try:
                A = build_adjacency(cfg_path)
            except Exception:
                A = np.eye(num_nodes, dtype=np.float32)
        else:
            A = np.eye(num_nodes, dtype=np.float32)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        self.layer1 = STGCNBlock(in_channels, 64, A)
        self.layer2 = STGCNBlock(64, 64, A)
        self.layer3 = STGCNBlock(64, 64, A)
        self.layer4 = STGCNBlock(64, 128, A)
        self.pool = nn.AdaptiveAvgPool2d((None,1))
        self.ctc_head = nn.Linear(128, num_class)
        self.nmm_head = nn.Linear(128, num_nmm) if num_nmm > 0 else None
        self.suffix_head = nn.Linear(128, num_suffix) if num_suffix > 0 else None
        self.rnm_head = nn.Linear(128, num_rnm) if num_rnm > 0 else None
        self.person_head = nn.Linear(128, num_person) if num_person > 0 else None
        self.number_head = nn.Linear(128, num_number) if num_number > 0 else None
        self.tense_head = nn.Linear(128, num_tense) if num_tense > 0 else None
        self.aspect_head = nn.Linear(128, num_aspect) if num_aspect > 0 else None
        self.mode_head = nn.Linear(128, num_mode) if num_mode > 0 else None

    def forward(self, x, return_features: bool = False):
        """Propagaci\u00f3n forward con opci\u00f3n de extraer features."""
        # x shape: (N, C, T, V)
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
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
