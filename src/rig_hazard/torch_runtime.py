from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPENDENCIES = PROJECT_ROOT / ".python_deps"
if LOCAL_DEPENDENCIES.exists() and str(LOCAL_DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPENDENCIES))

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Deep RIG-Hazard requires PyTorch and typing_extensions. "
        "Install requirements/deep_learning.txt before using deep commands."
    ) from exc


__all__ = ["torch", "nn", "Dataset"]
