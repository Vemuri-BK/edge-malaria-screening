"""
STEP 7 - EXPORT TO ONNX
=======================
Converts the trained PyTorch model to ONNX, the format the Raspberry Pi runs
with ONNX Runtime (no PyTorch needed on the Pi).

Input : "input"  float32 (N, 3, IMG_SIZE, IMG_SIZE), RGB in [0, 1]
Output: "logits" float32 (N, 2)  [Uninfected, Parasitized]

After export we check that ONNX and PyTorch give the same outputs.
"""
import numpy as np
import onnx
import onnxruntime as ort
import torch

from config import IMG_SIZE


def export_onnx(model, path, img_size: int = IMG_SIZE, opset: int = 17):
    model = model.cpu().eval()
    dummy = torch.rand(1, 3, img_size, img_size)
    kwargs = dict(input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                  opset_version=opset)
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:                     # older torch without the dynamo argument
        torch.onnx.export(model, dummy, str(path), **kwargs)
    onnx.checker.check_model(onnx.load(str(path)))
    return path


def check_parity(model, onnx_path, img_size: int = IMG_SIZE, n: int = 8) -> float:
    """Max absolute difference between PyTorch and ONNX logits on random inputs."""
    model = model.cpu().eval()
    x = torch.rand(n, 3, img_size, img_size)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    return float(np.abs(ref - out).max())
