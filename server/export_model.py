import argparse
import torch
from torch import nn


def main():
    p = argparse.ArgumentParser(description="Exportar modelo a TorchScript u ONNX")
    p.add_argument("--checkpoint", required=True, help="Ruta al checkpoint de PyTorch")
    p.add_argument("--output", required=True, help="Archivo de salida")
    p.add_argument("--onnx", action="store_true", help="Exportar en formato ONNX")
    p.add_argument("--seq_len", type=int, default=16, help="Longitud de secuencia simulada")
    args = p.parse_args()

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
