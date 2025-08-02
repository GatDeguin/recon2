#!/usr/bin/env python
import argparse
from pathlib import Path
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from models.shubert import SHuBERT

class RawH5Dataset(Dataset):
    """Load sequences of landmarks and optical flow from an HDF5 file."""
    def __init__(self, h5_path: str):
        self.h5 = h5py.File(h5_path, 'r')
        self.keys = list(self.h5.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        g = self.h5[self.keys[idx]]
        pose = g['pose'][:].reshape(-1, 33, 3)
        lh = g['left_hand'][:].reshape(-1, 21, 3)
        rh = g['right_hand'][:].reshape(-1, 21, 3)
        face = g['face'][:].reshape(-1, 468, 3)
        flow = g['optical_flow'][:]
        avg_flow = flow.mean(axis=(1,2))
        mag = np.linalg.norm(avg_flow, axis=-1, keepdims=True)
        flow_node = np.concatenate([avg_flow, mag], axis=1)
        nodes = np.concatenate([pose, lh, rh, face, flow_node[:, None, :]], axis=1)
        x = torch.from_numpy(nodes).permute(2,0,1).float()  # (C,T,V)
        return x

def collate_pretrain(batch):
    C, V = batch[0].shape[0], batch[0].shape[2]
    T = max(x.shape[1] for x in batch)
    padded = []
    lengths = []
    for x in batch:
        lengths.append(x.shape[1])
        if x.shape[1] < T:
            pad = torch.zeros(C, T - x.shape[1], V)
            x = torch.cat([x, pad], dim=1)
        padded.append(x)
    return torch.stack(padded), torch.tensor(lengths)

def train(args):
    ds = RawH5Dataset(args.h5_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pretrain)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SHuBERT(num_nodes=544)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for feats, lengths in dl:
            feats = feats.to(device)
            B, C, T, V = feats.shape
            lm_mask = torch.rand(B, T, device=device) < args.mask_prob
            flow_mask = torch.rand(B, T, device=device) < args.mask_prob
            out = model(feats.clone(), lm_mask, flow_mask)
            mse = (out - feats) ** 2
            lm_loss = mse[:,:,:,:-1] * lm_mask[:,None,:,None]
            flow_loss = mse[:,:,:,-1:] * flow_mask[:,None,:,None]
            loss = (lm_loss.sum() / lm_mask.sum().clamp(min=1)) + (flow_loss.sum() / flow_mask.sum().clamp(min=1))
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg = total_loss / len(dl)
        print(f"Epoch {epoch+1}: loss {avg:.4f}")
        torch.save(model.state_dict(), f'checkpoints/shubert_epoch{epoch+1}.pt')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pretrain SHuBERT with self-supervised masking')
    p.add_argument('--h5_file')
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch_size', type=int)
    p.add_argument('--mask_prob', type=float)
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parent / 'configs' / 'config.yaml'
    from utils.config import load_config, apply_defaults

    cfg = load_config(cfg_path)
    apply_defaults(args, cfg)

    train(args)
