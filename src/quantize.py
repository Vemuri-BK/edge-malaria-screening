"""
STEP 8 - INT8 QUANTIZATION
==========================
Post-training STATIC quantization with ONNX Runtime.

    FP32 weights/activations  ->  INT8   (~4x smaller, faster on ARM CPUs)

Static = activation ranges are measured once on a small calibration set of
training images. (Dynamic quantization would leave the convolutions in FP32
on ONNX Runtime's CPU backend, giving almost no speed-up for a CNN.)

Format: QDQ (QuantizeLinear/DequantizeLinear nodes), per-channel weights.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

from config import IMG_SIZE
from dataset import preprocess_pil


class ImageCalibrationReader(CalibrationDataReader):
    """Feeds calibration images one at a time to the quantizer."""

    def __init__(self, image_paths, img_size: int = IMG_SIZE):
        self.paths = list(image_paths)
        self.img_size = img_size
        self._it = iter(self.paths)

    def get_next(self):
        p = next(self._it, None)
        if p is None:
            return None
        with Image.open(p) as img:
            x = preprocess_pil(img, self.img_size)[None]      # (1, 3, H, W)
        return {"input": x}

    def rewind(self):
        self._it = iter(self.paths)


class ArrayCalibrationReader(CalibrationDataReader):
    """Same as ImageCalibrationReader, but from an in-memory float32 array (N, 3, H, W)."""

    def __init__(self, X: np.ndarray):
        self.X = X
        self.i = 0

    def get_next(self):
        if self.i >= len(self.X):
            return None
        x = self.X[self.i:self.i + 1]
        self.i += 1
        return {"input": x}

    def rewind(self):
        self.i = 0


def quantize_int8(fp32_path, int8_path, calib, method: str = "minmax", nodes_to_exclude=None):
    """
    calib  : list of image paths OR float32 array (N, 3, H, W)
    method : "minmax" | "percentile" | "entropy"
             (percentile ignores extreme activations - much better for MobileNetV3)
    nodes_to_exclude: ONNX node names to keep in FP32 (mixed precision)
    """
    fp32_path, int8_path = Path(fp32_path), Path(int8_path)

    # Recommended pre-processing (shape inference + graph optimization)
    prep_path = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        quant_pre_process(str(fp32_path), str(prep_path))
        src = prep_path
    except Exception as e:
        print(f"[warn] quant_pre_process skipped ({e})")
        src = fp32_path

    calib_method = {"minmax": CalibrationMethod.MinMax,
                    "percentile": CalibrationMethod.Percentile,
                    "entropy": CalibrationMethod.Entropy}[method]

    quantize_static(
        model_input=str(src),
        model_output=str(int8_path),
        calibration_data_reader=(ArrayCalibrationReader(calib) if isinstance(calib, np.ndarray)
                                 else ImageCalibrationReader(calib)),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=calib_method,
        nodes_to_exclude=nodes_to_exclude or [],
    )
    if prep_path.exists():
        prep_path.unlink()
    return int8_path
