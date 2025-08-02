import argparse
from pathlib import Path
import torch
from torch import nn


def main():
    p = argparse.ArgumentParser(description="Exportar modelo a TorchScript u ONNX")
    p.add_argument("--checkpoint")
    p.add_argument("--output")
    p.add_argument("--onnx", action="store_true", help="Exportar en formato ONNX")
    p.add_argument("--seq_len", type=int)
    args = p.parse_args()

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    from utils.config import load_config, apply_defaults

    cfg = load_config(cfg_path)
    apply_defaults(args, cfg)

    model = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(model, nn.Module):
        model.eval()
    dummy = torch.zeros(1, 3, args.seq_len, 544)
    if args.onnx:
        torch.onnx.export(model, dummy, args.output, opset_version=12)
    else:
        traced = torch.jit.trace(model, dummy)
        traced.save(args.output)
    print(f"Modelo exportado a {args.output}")


if __name__ == "__main__":
    main()
