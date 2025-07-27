#!/usr/bin/env python
import os
import argparse
import h5py
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from metrics import MetricsLogger
from models.stgcn import STGCN
from models.sttn import STTN
from models.corrnet import CorrNetPlus
from models.mcst_transformer import MCSTTransformer

class SignDataset(Dataset):
    def __init__(
        self,
        h5_path,
        csv_path,
        domain_csv=None,
        segments: bool = False,
        include_openface: bool = False,
        augment=None,
    ):
        """Dataset for sign language graph sequences.

        Parameters
        ----------
        h5_path : str
            Ruta al archivo ``HDF5`` con las secuencias.
        csv_path : str
            Archivo ``CSV`` con las etiquetas.
        domain_csv : str, optional
            CSV con dominios para DANN.
        segments : bool, default ``False``
            Leer segmentos pre-cortados.
        include_openface : bool, default ``False``
            Incluir nodos adicionales de OpenFace.
        augment : callable, optional
            Función que recibe ``x`` y retorna la versión aumentada.
        """

        self.h5 = h5py.File(h5_path, 'r')
        self.include_openface = include_openface
        self.augment = augment
        df = pd.read_csv(csv_path, sep=';')
        label_map = {str(r['id']): str(r['label']) for _, r in df.iterrows()}
        nmm_map = {str(r['id']): str(r.get('nmm', 'none')) for _, r in df.iterrows()} if 'nmm' in df.columns else {}
        suf_map = {str(r['id']): str(r.get('suffix', 'none')) for _, r in df.iterrows()} if 'suffix' in df.columns else {}
        rnm_map = {str(r['id']): str(r.get('rnm', 'none')) for _, r in df.iterrows()} if 'rnm' in df.columns else {}
        per_map = {str(r['id']): str(r.get('person', 'none')) for _, r in df.iterrows()} if 'person' in df.columns else {}
        num_map = {str(r['id']): str(r.get('number', 'none')) for _, r in df.iterrows()} if 'number' in df.columns else {}
        tense_map = {str(r['id']): str(r.get('tense', 'none')) for _, r in df.iterrows()} if 'tense' in df.columns else {}
        aspect_map = {str(r['id']): str(r.get('aspect', 'none')) for _, r in df.iterrows()} if 'aspect' in df.columns else {}
        mode_map = {str(r['id']): str(r.get('mode', 'none')) for _, r in df.iterrows()} if 'mode' in df.columns else {}
        self.domain_map = {}
        if domain_csv:
            dom = pd.read_csv(domain_csv, sep=';')
            self.domain_map = {str(r['id']): int(r['domain']) for _, r in dom.iterrows()}
        self.samples = []

        def _add_sample(path: str, base_id: str) -> None:
            if base_id in label_map:
                domain = self.domain_map.get(base_id, 0)
                nmm = nmm_map.get(base_id, 'none') if nmm_map else 'none'
                suf = suf_map.get(base_id, 'none') if suf_map else 'none'
                rnm = rnm_map.get(base_id, 'none') if rnm_map else 'none'
                per = per_map.get(base_id, 'none') if per_map else 'none'
                num = num_map.get(base_id, 'none') if num_map else 'none'
                tense = tense_map.get(base_id, 'none') if tense_map else 'none'
                aspect = aspect_map.get(base_id, 'none') if aspect_map else 'none'
                mode = mode_map.get(base_id, 'none') if mode_map else 'none'
                self.samples.append(
                    (path, label_map[base_id], domain, nmm, suf, rnm, per, num, tense, aspect, mode),
                )

        for vid in self.h5.keys():
            base = os.path.splitext(vid)[0]
            segs = []
            if segments:
                grp = self.h5[vid]
                segs = sorted(k for k in grp.keys() if k.startswith('segment_'))
            if segments and segs:
                for seg in segs:
                    _add_sample(f"{vid}/{seg}", base)
            else:
                _add_sample(vid, base)
        self.vocab = self._build_vocab()
        self.nmm_vocab = self._build_map([s[3] for s in self.samples])
        self.suffix_vocab = self._build_map([s[4] for s in self.samples])
        self.rnm_vocab = self._build_map([s[5] for s in self.samples])
        self.person_vocab = self._build_map([s[6] for s in self.samples])
        self.number_vocab = self._build_map([s[7] for s in self.samples])
        self.tense_vocab = self._build_map([s[8] for s in self.samples])
        self.aspect_vocab = self._build_map([s[9] for s in self.samples])
        self.mode_vocab = self._build_map([s[10] for s in self.samples])
        self.num_domains = len(set(s[2] for s in self.samples)) or 1

        # Determine number of graph nodes
        self.num_nodes = 544
        self.n_aus = 0
        if self.include_openface and self.samples:
            g = self.h5[self.samples[0][0]]
            if 'head_pose' in g:
                self.num_nodes += 1
            if 'torso_pose' in g:
                self.num_nodes += 1
            if 'aus' in g:
                self.n_aus = g['aus'].shape[1]
                self.num_nodes += self.n_aus

    def close(self):
        """Close the underlying HDF5 file if open."""
        if getattr(self, "h5", None) is not None:
            try:
                self.h5.close()
            finally:
                self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()

    def _build_vocab(self):
        tokens = set()
        for s in self.samples:
            tokens.update(s[1].split())
        vocab = {
            '<blank>': 0,
            '<sos>': 1,
            '<eos>': 2,
        }
        next_index = 3
        for i, tok in enumerate(sorted(tokens)):
            vocab[tok] = next_index + i
        return vocab

    @staticmethod
    def _build_map(items):
        uniq = sorted(set(items))
        return {v: i for i, v in enumerate(uniq)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, lbl, domain, nmm, suf, rnm, per, num, tense, aspect, mode = self.samples[idx]
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
        if self.include_openface:
            if 'head_pose' in g:
                head = g['head_pose'][:].reshape(-1, 1, 3)
                nodes = np.concatenate([nodes, head], axis=1)
            if 'torso_pose' in g:
                torso = g['torso_pose'][:].reshape(-1, 1, 3)
                nodes = np.concatenate([nodes, torso], axis=1)
            if 'aus' in g:
                aus = g['aus'][:]
                aus = np.repeat(aus[:, :, None], 3, axis=2)
                nodes = np.concatenate([nodes, aus], axis=1)
        x = torch.from_numpy(nodes).permute(2, 0, 1).float()  # (C,T,V)
        if self.augment is not None:
            x = self.augment(x)
        tokens = [self.vocab[t] for t in lbl.split() if t in self.vocab]
        y = torch.tensor(tokens, dtype=torch.long)
        nmm_id = self.nmm_vocab[nmm]
        suf_id = self.suffix_vocab[suf]
        rnm_id = self.rnm_vocab[rnm]
        per_id = self.person_vocab[per]
        num_id = self.number_vocab[num]
        tense_id = self.tense_vocab[tense]
        aspect_id = self.aspect_vocab[aspect]
        mode_id = self.mode_vocab[mode]
        return (
            x,
            y,
            domain,
            nmm_id,
            suf_id,
            rnm_id,
            per_id,
            num_id,
            tense_id,
            aspect_id,
            mode_id,
        )

def collate(batch):
    """Pad secuencias y empaquetar dominios y labels auxiliares."""
    (
        feats,
        labels,
        domains,
        nmms,
        sufs,
        rnms,
        pers,
        nums,
        tenses,
        aspects,
        modes,
    ) = zip(*batch)
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
    doms = torch.tensor(domains, dtype=torch.long)
    nmm_t = torch.tensor(nmms, dtype=torch.long)
    suf_t = torch.tensor(sufs, dtype=torch.long)
    rnm_t = torch.tensor(rnms, dtype=torch.long)
    per_t = torch.tensor(pers, dtype=torch.long)
    num_t = torch.tensor(nums, dtype=torch.long)
    tense_t = torch.tensor(tenses, dtype=torch.long)
    aspect_t = torch.tensor(aspects, dtype=torch.long)
    mode_t = torch.tensor(modes, dtype=torch.long)
    return (
        padded_feats,
        padded_labels,
        torch.tensor(feat_lengths),
        torch.tensor(label_lengths),
        doms,
        nmm_t,
        suf_t,
        rnm_t,
        per_t,
        num_t,
        tense_t,
        aspect_t,
        mode_t,
    )

def _contrastive(feats: torch.Tensor) -> torch.Tensor:
    """Simple NT-Xent style loss over batch-averaged features."""
    f = feats.mean(dim=1)
    f = nn.functional.normalize(f, dim=1)
    sim = torch.matmul(f, f.t()) / 0.1
    labels = torch.arange(f.size(0), device=f.device)
    return nn.functional.cross_entropy(sim, labels)

def build_model(
    name: str,
    num_classes: int,
    num_nmm: int = 0,
    num_suffix: int = 0,
    num_rnm: int = 0,
    num_person: int = 0,
    num_number: int = 0,
    num_tense: int = 0,
    num_aspect: int = 0,
    num_mode: int = 0,
    num_nodes: int = 544,
) -> nn.Module:
    """Create the selected model."""
    if name == 'stgcn':
        return STGCN(
            in_channels=3,
            num_class=num_classes,
            num_nodes=num_nodes,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
            num_person=num_person,
            num_number=num_number,
            num_tense=num_tense,
            num_aspect=num_aspect,
            num_mode=num_mode,
        )
    if name == 'sttn':
        return STTN(
            in_channels=3,
            num_class=num_classes,
            num_nodes=num_nodes,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
            num_person=num_person,
            num_number=num_number,
            num_tense=num_tense,
            num_aspect=num_aspect,
            num_mode=num_mode,
        )
    if name == 'corrnet+':
        return CorrNetPlus(
            in_channels=3,
            num_class=num_classes,
            num_nodes=num_nodes,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
            num_person=num_person,
            num_number=num_number,
            num_tense=num_tense,
            num_aspect=num_aspect,
            num_mode=num_mode,
        )
    if name == 'mcst':
        return MCSTTransformer(
            in_channels=3,
            num_class=num_classes,
            num_nodes=num_nodes,
            num_nmm=num_nmm,
            num_suffix=num_suffix,
            num_rnm=num_rnm,
            num_person=num_person,
            num_number=num_number,
            num_tense=num_tense,
            num_aspect=num_aspect,
            num_mode=num_mode,
        )
    raise ValueError(f'Unknown model: {name}')


def evaluate(model: nn.Module, dl: DataLoader, inv_vocab: dict, device: torch.device) -> Tuple[float, float]:
    """Compute simple WER and NMM accuracy on the provided dataloader."""
    try:
        from jiwer import wer
    except Exception:
        wer = None
    model.eval()
    total_words = 0
    word_errs = 0
    correct_nmm = 0
    count_nmm = 0
    with torch.no_grad():
        for batch in dl:
            (
                feats,
                labels,
                feat_lens,
                label_lens,
                domains,
                nmm_lbls,
                *_,
            ) = batch
            feats = feats.to(device)
            labels = labels.to(device)
            nmm_lbls = nmm_lbls.to(device)
            logits, nmm_logits, _ = model(feats)
            preds = logits.argmax(-1)
            for p, t in zip(preds, labels):
                pred_tokens = []
                last = 0
                skip = {0, 1, 2}
                for tok in p.tolist():
                    if tok not in skip and tok != last:
                        pred_tokens.append(inv_vocab.get(tok, ""))
                    last = tok
                tgt_tokens = [inv_vocab.get(int(x), "") for x in t if int(x) not in skip]
                hyp = " ".join(pred_tokens).strip()
                ref = " ".join(tgt_tokens).strip()
                if wer:
                    word_errs += wer(ref, hyp) * len(ref.split())
                total_words += len(ref.split())
            if nmm_logits is not None:
                pred_nmm = nmm_logits.argmax(-1)
                correct_nmm += (pred_nmm.cpu() == nmm_lbls.cpu()).sum().item()
                count_nmm += len(nmm_lbls)
    wer_val = word_errs / total_words if total_words else 0.0
    nmm_acc = correct_nmm / count_nmm if count_nmm else 0.0
    return wer_val, nmm_acc


def train(args):
    with SignDataset(
        args.h5_file,
        args.csv_file,
        args.domain_labels,
        segments=args.segments,
        include_openface=args.include_openface,
    ) as ds:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(
            args.model,
            len(ds.vocab),
            len(ds.nmm_vocab),
            len(ds.suffix_vocab),
            len(ds.rnm_vocab),
            len(ds.person_vocab),
            len(ds.number_vocab),
            len(ds.tense_vocab),
            len(ds.aspect_vocab),
            len(ds.mode_vocab),
            num_nodes=ds.num_nodes,
        )
        model.to(device)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        ce_loss = nn.CrossEntropyLoss()
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
        logger = MetricsLogger(os.path.join('logs', 'metrics.db'))

        inv_vocab = {v: k for k, v in ds.vocab.items()}

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            for batch in dl:
                (
                    feats,
                    labels,
                    feat_lens,
                    label_lens,
                    domains,
                    nmm_lbls,
                    suf_lbls,
                    rnm_lbls,
                    per_lbls,
                    num_lbls,
                    tense_lbls,
                    aspect_lbls,
                    mode_lbls,
                ) = batch
                domains = domains.to(device)
                feats = feats.to(device)
                labels = labels.to(device)
                nmm_lbls = nmm_lbls.to(device)
                suf_lbls = suf_lbls.to(device)
                rnm_lbls = rnm_lbls.to(device)
                per_lbls = per_lbls.to(device)
                num_lbls = num_lbls.to(device)
                tense_lbls = tense_lbls.to(device)
                aspect_lbls = aspect_lbls.to(device)
                mode_lbls = mode_lbls.to(device)
                feat_lens = feat_lens.to(device)
                label_lens = label_lens.to(device)
                return_feat = args.domain_labels or args.contrastive
                if return_feat:
                    (
                        gloss_logits,
                        nmm_logits,
                        suf_logits,
                        rnm_logits,
                        per_logits,
                        num_logits,
                        tense_logits,
                        aspect_logits,
                        mode_logits,
                    ), feats_emb = model(feats, return_features=True)
                else:
                    (
                        gloss_logits,
                        nmm_logits,
                        suf_logits,
                        rnm_logits,
                        per_logits,
                        num_logits,
                        tense_logits,
                        aspect_logits,
                        mode_logits,
                    ) = model(feats)
                outputs = gloss_logits.permute(1, 0, 2)  # T,B,C
                # nn.CTCLoss requiere que todas las etiquetas estén
                # concatenadas en un solo vector y se pasen sus longitudes
                # originales. No quitar flatten() ni label_lens al extender
                # el entrenamiento.
                targets = labels.flatten()
                loss = criterion(outputs, targets, feat_lens, label_lens)
                if nmm_logits is not None:
                    loss = loss + ce_loss(nmm_logits, nmm_lbls)
                if suf_logits is not None:
                    loss = loss + ce_loss(suf_logits, suf_lbls)
                if rnm_logits is not None:
                    loss = loss + ce_loss(rnm_logits, rnm_lbls)
                if per_logits is not None:
                    loss = loss + ce_loss(per_logits, per_lbls)
                if num_logits is not None:
                    loss = loss + ce_loss(num_logits, num_lbls)
                if tense_logits is not None:
                    loss = loss + ce_loss(tense_logits, tense_lbls)
                if aspect_logits is not None:
                    loss = loss + ce_loss(aspect_logits, aspect_lbls)
                if mode_logits is not None:
                    loss = loss + ce_loss(mode_logits, mode_lbls)
                if args.domain_labels:
                    dom_feat = feats_emb.mean(dim=1)
                    dom_logits = disc(grad_reverse(dom_feat))
                    adv_loss = adv_criterion(dom_logits, domains)
                    loss = loss + 0.1 * adv_loss
                if args.contrastive:
                    cont = _contrastive(feats_emb)
                    loss = loss + 0.1 * cont
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

            wer_val, nmm_acc = evaluate(model, dl, inv_vocab, device)
            logger.log(wer=wer_val, nmm_acc=nmm_acc)

        logger.close()
        with open('vocab.txt', 'w', encoding='utf-8') as f:
            for i in range(len(inv_vocab)):
                f.write(inv_vocab.get(i, '') + '\n')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train sign language models with CTC loss')
    p.add_argument('--h5_file', required=True, help='HDF5 file with landmarks and optical flow')
    p.add_argument('--csv_file', required=True, help='CSV file with transcripts')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--model', type=str, default='stgcn', choices=['stgcn', 'sttn', 'corrnet+', 'mcst'], help='Model architecture')
    p.add_argument('--domain_labels', help='CSV con etiquetas de dominio opcional')
    p.add_argument('--contrastive', action='store_true', help='Use contrastive loss')
    p.add_argument('--segments', action='store_true', help='Load segment_* subgroups as separate samples')
    p.add_argument('--include_openface', action='store_true', help='Load head/torso pose and AUs if present')
    args = p.parse_args()
    train(args)
