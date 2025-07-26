import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self, in_channels, num_class, num_nodes):
        super().__init__()
        A = np.eye(num_nodes, dtype=np.float32)
        self.data_bn = nn.BatchNorm1d(num_nodes * in_channels)
        self.layer1 = STGCNBlock(in_channels, 64, A)
        self.layer2 = STGCNBlock(64, 64, A)
        self.layer3 = STGCNBlock(64, 64, A)
        self.layer4 = STGCNBlock(64, 128, A)
        self.pool = nn.AdaptiveAvgPool2d((None,1))
        self.fc = nn.Linear(128, num_class)

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
        out = self.fc(feat)
        log_probs = out.log_softmax(-1)
        if return_features:
            return log_probs, feat
        return log_probs
