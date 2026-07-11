"""GPU capability helpers.

These utilities let the training pipeline run unchanged on GPUs that do not
support bfloat16 (e.g. the Tesla T4 on Kaggle, compute capability 7.5), by
transparently falling back to fp16 for both the model weights and the trainer's
mixed-precision setting.
"""
import os
from typing import Optional, Tuple

import torch


def is_kaggle() -> bool:
    """Return True when running inside a Kaggle kernel."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_gpu_compute_capability() -> Optional[Tuple[int, int]]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def supports_bfloat16() -> bool:
    """Native bf16 tensor cores exist on Ampere (SM 8.0) and newer."""
    cap = get_gpu_compute_capability()
    if cap is None:
        return False
    return cap[0] >= 8


def get_model_dtype() -> torch.dtype:
    """Weight dtype to load the backbone with.

    bf16 on capable GPUs; float32 elsewhere so that fp16 AMP (with a
    GradScaler) stays numerically stable on Turing GPUs like the T4.
    """
    return torch.bfloat16 if supports_bfloat16() else torch.float32


def get_trainer_precision(config_precision):
    """Downgrade a bf16 precision setting to fp16 when bf16 is unsupported.

    Returns a value valid for PyTorch Lightning 1.9.x
    (``16`` for fp16 mixed precision, ``"bf16"`` for bf16 mixed precision).
    """
    precision = str(config_precision)
    if precision in ("bf16", "bf16-mixed", "16-mixed") and not supports_bfloat16():
        return 16
    if precision in ("16-mixed",):
        return 16
    return config_precision


def describe_gpu() -> str:
    if not torch.cuda.is_available():
        return "CPU (CUDA not available)"
    count = torch.cuda.device_count()
    names = {torch.cuda.get_device_name(i) for i in range(count)}
    cap = get_gpu_compute_capability()
    cap_str = f"{cap[0]}.{cap[1]}" if cap else "unknown"
    dtype = "bf16" if supports_bfloat16() else "fp16"
    return f"{count} x {', '.join(sorted(names))} (compute {cap_str}, using {dtype})"
