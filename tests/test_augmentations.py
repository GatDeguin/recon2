import torch
from augmentations import temporal_mixup, speed_perturbation, overlay_gan_background


def test_temporal_mixup_deterministic():
    torch.manual_seed(0)
    a = torch.ones(3, 4, 2)
    b = torch.zeros(3, 4, 2)
    out1 = temporal_mixup(a, b, alpha=0.5)
    torch.manual_seed(0)
    out2 = temporal_mixup(a, b, alpha=0.5)
    assert torch.allclose(out1, out2)
    assert out1.shape == a.shape


def test_speed_perturbation_shape():
    x = torch.randn(3, 4, 2)
    out = speed_perturbation(x, rate=1.5)
    assert out.shape[0] == 3
    assert out.shape[2] == 2
    assert out.shape[1] == int(round(4 * 1.5))


def test_overlay_gan_background():
    def dummy_gan(inp: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(1)
        return torch.rand_like(inp) * 0.1

    torch.manual_seed(1)
    x = torch.zeros(3, 4, 2)
    out1 = overlay_gan_background(x, dummy_gan)
    torch.manual_seed(1)
    out2 = overlay_gan_background(x, dummy_gan)
    assert torch.allclose(out1, out2)
    assert out1.shape == x.shape
