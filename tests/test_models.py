import torch
from train import build_model


def _run_forward(name: str):
    model = build_model(name, num_classes=5)
    inp = torch.randn(2, 3, 4, 544)
    out = model(inp)
    assert out.shape == (2, 4, 5)


def test_stgcn_forward():
    _run_forward("stgcn")


def test_sttn_forward():
    _run_forward("sttn")


def test_corrnet_forward():
    _run_forward("corrnet+")


def test_mcst_forward():
    _run_forward("mcst")
