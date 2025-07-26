#!/usr/bin/env python
import os
import argparse
import h5py
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.stgcn import STGCN

class SignDataset(Dataset):
    def __init__(self, h5_path, csv_path):
        self.h5 = h5py.File(h5_path, 'r')
        df = pd.read_csv(csv_path, sep=';')
        label_map = {str(r['id']): str(r['label']) for _, r in df.iterrows()}
        self.samples = []
        for vid in self.h5.keys():
            base = os.path.splitext(vid)[0]
            if base in label_map:
                self.samples.append((vid, label_map[base]))
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        tokens = set()
        for _, lbl in self.samples:
            tokens.update(lbl.split())
        vocab = {tok: i+1 for i, tok in enumerate(sorted(tokens))}
        vocab['<blank>'] = 0
        return vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, lbl = self.samples[idx]
        g = self.h5[vid]
        pose = g['pose'][:].reshape(-1, 33, 3)
        lh = g['left_hand'][:].reshape(-1, 21, 3)
        rh = g['right_hand'][:].reshape(-1, 21, 3)
        face = g['face'][:].reshape(-1, 468, 3)
        nodes = np.concatenate([pose, lh, rh, face], axis=1)
        flow = g['optical_flow'][:]
        avg_flow = flow.mean(axis=(1,2))  # (T,2)
        mag = np.linalg.norm(avg_flow, axis=-1, keepdims=True)
        flow_node = np.concatenate([avg_flow, mag], axis=1)  # (T,3)
        nodes = np.concatenate([nodes, flow_node[:, None, :]], axis=1)
        x = torch.from_numpy(nodes).permute(2,0,1).float()  # (C,T,V)
        tokens = [self.vocab[t] for t in lbl.split() if t in self.vocab]
        y = torch.tensor(tokens, dtype=torch.long)
        return x, y

def collate(batch):
    feats, labels = zip(*batch)
    T = max(f.shape[1] for f in feats)
    V = feats[0].shape[2]
    C = feats[0].shape[0]
    padded_feats = []
    feat_lengths = []
    for f in feats:
        feat_lengths.append(f.shape[1])
        if f.shape[1] < T:
            pad = torch.zeros(C, T - f.shape[1], V)
            f = torch.cat([f, pad], dim=1)
        padded_feats.append(f)
    padded_feats = torch.stack(padded_feats)
    L = max(len(l) for l in labels)
    padded_labels = []
    label_lengths = []
    for l in labels:
        label_lengths.append(len(l))
        if len(l) < L:
            pad = torch.zeros(L - len(l), dtype=torch.long)
            l = torch.cat([l, pad])
        padded_labels.append(l)
    padded_labels = torch.stack(padded_labels)
    return padded_feats, padded_labels, torch.tensor(feat_lengths), torch.tensor(label_lengths)

def train(args):
    ds = SignDataset(args.h5_file, args.csv_file)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = STGCN(in_channels=3, num_class=len(ds.vocab), num_nodes=544)
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for feats, labels, feat_lens, label_lens in dl:
            feats = feats.to(device)
            labels = labels.to(device)
            outputs = model(feats)
            outputs = outputs.permute(1,0,2)  # T,B,C
            loss = criterion(outputs, labels, feat_lens, label_lens)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(dl)
        print(f"Epoch {epoch+1}: loss {avg:.4f}")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'vocab': ds.vocab
        }, f'checkpoints/epoch_{epoch+1}.pt')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train ST-GCN with CTC loss')
    p.add_argument('--h5_file', required=True, help='HDF5 file with landmarks and optical flow')
    p.add_argument('--csv_file', required=True, help='CSV file with transcripts')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    args = p.parse_args()
    train(args)
