import argparse
import torch
from torch import nn

from train import build_model


def load_model(checkpoint: str, arch: str) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        model = ckpt
        model.eval()
        return model
    vocab = ckpt.get("vocab")
    if vocab is None:
        raise ValueError("Checkpoint debe incluir vocabulario o ser un Module")
    model = build_model(arch, len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser(
        description="Exportar checkpoint a ONNX y cuantizar dinamicamente"
    )
    p.add_argument("--checkpoint", required=True, help="Ruta al checkpoint")
    p.add_argument("--arch", default="stgcn",
                   choices=["stgcn", "sttn", "corrnet+", "mcst"],
                   help="Arquitectura del modelo")
    p.add_argument("--output", required=True, help="Archivo onnx de salida")
    p.add_argument("--seq_len", type=int, default=16,
                   help="Longitud de secuencia dummy")
    args = p.parse_args()

    model = load_model(args.checkpoint, args.arch)
    dummy = torch.zeros(1, 3, args.seq_len, 544)

    dynamic = {0: "batch", 2: "time"}
    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=12,
        input_names=["input"],
        dynamic_axes={"input": dynamic}
    )
    print(f"ONNX guardado en {args.output}")

    quant_path = args.output.replace(".onnx", "_int8.onnx")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(args.output, quant_path, weight_type=QuantType.QInt8)
        print(f"Modelo cuantizado en {quant_path}")
    except Exception as e:
        print("Cuantizacion fallida:", e)
        return

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
        outputs = sess.run(None, {"input": dummy.numpy()})
        print("Inferencia correcta, shapes:", [o.shape for o in outputs])
    except Exception as e:
        print("Error al ejecutar inferencia en onnxruntime:", e)


if __name__ == "__main__":
    main()
