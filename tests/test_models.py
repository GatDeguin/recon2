import torch
from train import build_model


def _run_forward(name: str):
    model = build_model(name, num_classes=5, num_nmm=2, num_suffix=3)
    inp = torch.randn(2, 3, 4, 544)
    gloss, nmm, suf = model(inp)
    assert gloss.shape == (2, 4, 5)
    assert nmm.shape == (2, 2)
    assert suf.shape == (2, 3)


def test_stgcn_forward():
    _run_forward("stgcn")


def test_sttn_forward():
    _run_forward("sttn")


def test_corrnet_forward():
    _run_forward("corrnet+")


def test_mcst_forward():
    _run_forward("mcst")
