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
from models.sttn import STTN
from models.corrnet import CorrNetPlus

class SignDataset(Dataset):
    def __init__(self, h5_path, csv_path, domain_csv=None):
        self.h5 = h5py.File(h5_path, 'r')
        df = pd.read_csv(csv_path, sep=';')
        label_map = {str(r['id']): str(r['label']) for _, r in df.iterrows()}
        self.domain_map = {}
        if domain_csv:
            dom = pd.read_csv(domain_csv, sep=';')
            self.domain_map = {str(r['id']): int(r['domain']) for _, r in dom.iterrows()}
        self.samples = []
        for vid in self.h5.keys():
            base = os.path.splitext(vid)[0]
            if base in label_map:
                domain = self.domain_map.get(base, 0)
                self.samples.append((vid, label_map[base], domain))
        self.vocab = self._build_vocab()
        self.num_domains = len(set(d for _, _, d in self.samples)) or 1

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
        vid, lbl, domain = self.samples[idx]
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
        return x, y, domain

def collate(batch):
    """Pad secuencias y empaquetar dominios opcionalmente.

    El tensor de etiquetas resultante tiene forma ``(B, L)`` y debe
    aplanarse (`flatten`) junto con ``label_lengths`` antes de pasar a
    ``nn.CTCLoss``.
    """
    if len(batch[0]) == 3:
        feats, labels, domains = zip(*batch)
    else:
        feats, labels = zip(*batch)
        domains = None
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
    if domains is not None:
        doms = torch.tensor(domains, dtype=torch.long)
        return padded_feats, padded_labels, torch.tensor(feat_lengths), torch.tensor(label_lengths), doms
    return padded_feats, padded_labels, torch.tensor(feat_lengths), torch.tensor(label_lengths)

def build_model(name: str, num_classes: int) -> nn.Module:
    """Create the selected model."""
    if name == 'stgcn':
        return STGCN(in_channels=3, num_class=num_classes, num_nodes=544)
    if name == 'sttn':
        return STTN(in_channels=3, num_class=num_classes, num_nodes=544)
    if name == 'corrnet+':
        return CorrNetPlus(in_channels=3, num_class=num_classes, num_nodes=544)
    raise ValueError(f'Unknown model: {name}')


def train(args):
    ds = SignDataset(args.h5_file, args.csv_file, args.domain_labels)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(args.model, len(ds.vocab))
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    if args.domain_labels:
        from dann import DomainDiscriminator, grad_reverse
        disc = DomainDiscriminator(128, ds.num_domains)
        disc.to(device)
        disc_optim = torch.optim.Adam(disc.parameters(), lr=1e-3)
        adv_criterion = nn.CrossEntropyLoss()
    else:
        disc = disc_optim = adv_criterion = None
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in dl:
            if args.domain_labels:
                feats, labels, feat_lens, label_lens, domains = batch
                domains = domains.to(device)
            else:
                feats, labels, feat_lens, label_lens = batch
            feats = feats.to(device)
            labels = labels.to(device)
            feat_lens = feat_lens.to(device)
            label_lens = label_lens.to(device)
            if args.domain_labels:
                outputs, feats_emb = model(feats, return_features=True)
            else:
                outputs = model(feats)
            outputs = outputs.permute(1, 0, 2)  # T,B,C
            # nn.CTCLoss requiere que todas las etiquetas estén
            # concatenadas en un solo vector y se pasen sus longitudes
            # originales. No quitar flatten() ni label_lens al extender
            # el entrenamiento.
            targets = labels.flatten()
            loss = criterion(outputs, targets, feat_lens, label_lens)
            if args.domain_labels:
                dom_feat = feats_emb.mean(dim=1)
                dom_logits = disc(grad_reverse(dom_feat))
                adv_loss = adv_criterion(dom_logits, domains)
                loss = loss + 0.1 * adv_loss
            optim.zero_grad()
            if disc_optim:
                disc_optim.zero_grad()
            loss.backward()
            optim.step()
            if disc_optim:
                disc_optim.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(dl)
        print(f"Epoch {epoch+1}: loss {avg:.4f}")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optim.state_dict(),
            'vocab': ds.vocab
        }, f'checkpoints/epoch_{epoch+1}.pt')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train sign language models with CTC loss')
    p.add_argument('--h5_file', required=True, help='HDF5 file with landmarks and optical flow')
    p.add_argument('--csv_file', required=True, help='CSV file with transcripts')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--model', type=str, default='stgcn', choices=['stgcn', 'sttn', 'corrnet+'], help='Model architecture')
    p.add_argument('--domain_labels', help='CSV con etiquetas de dominio opcional')
    args = p.parse_args()
    train(args)
