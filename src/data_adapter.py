"""
data_adapter.py — h5 → Braindecode data loader + model factory

Usage (Colab):
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/ML_project/src")
    from data_adapter import load_dataset, create_model, get_dataloaders

Usage (local):
    from data_adapter import load_dataset, create_model, get_dataloaders
"""

import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class H5Dataset(Dataset):
    """Load pre-processed h5 data for Braindecode models.

    Each sample is shape (channels, time).  The dataset does NOT add a
    singleton channel-dimension — that is handled by BraindecodeModelWrapper
    so the raw arrays stay model-agnostic.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray | None = None):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]


# ═══════════════════════════════════════════════════════════════
# Data loader
# ═══════════════════════════════════════════════════════════════

def load_dataset(data_root: str | Path, dataset_name: str):
    """Load train / val / test splits from pre-processed h5 files.

    Parameters
    ----------
    data_root : path to the ``course project`` folder that contains one
                sub-directory per dataset.
    dataset_name : one of ``BCIC2A``, ``CHINESE``, ``MDD``, ``SEED``,
                   ``SLEEP``.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, metadata
    """
    base = Path(data_root) / dataset_name

    # ---- dataset metadata ----
    info_path = base / "dataset_info.json"
    if not info_path.exists():
        info_path = base / "dataset_info_fixed.json"
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)

    categories: list[str] = info["dataset"]["category_list"]
    num_classes: int = len(categories)
    sfreq: float = info["processing"]["target_sampling_rate"]

    # ---- load h5 arrays ----
    def _load_h5(path: Path):
        with h5py.File(str(path), "r") as f:
            keys = list(f.keys())
            X = f["X"][()].astype(np.float32)
            y = f["y"][()].astype(np.int64) if "y" in keys else None
        return X, y

    X_train, y_train = _load_h5(base / "train.h5")
    X_val,   y_val   = _load_h5(base / "val.h5")
    X_test,  _       = _load_h5(base / "test_x_only.h5")

    n_channels, n_time = X_train.shape[1], X_train.shape[2]

    metadata = {
        "name":          dataset_name,
        "categories":    categories,
        "num_classes":   num_classes,
        "n_channels":    n_channels,
        "n_time":        n_time,
        "sfreq":         sfreq,
        "window_sec":    n_time / sfreq,
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_test":        len(X_test),
    }
    return X_train, y_train, X_val, y_val, X_test, metadata


def get_dataloaders(X_train, y_train, X_val, y_val, X_test,
                    batch_size: int = 32):
    """Standard PyTorch DataLoaders.

    Test loader uses batch_size=1 (order must match course requirement).
    """
    train_loader = DataLoader(
        H5Dataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(
        H5Dataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(
        H5Dataset(X_test),           batch_size=1,          shuffle=False)
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════
# Model wrapper
# ═══════════════════════════════════════════════════════════════

class BraindecodeModelWrapper(nn.Module):
    """Keep a stable local wrapper around Braindecode models.

    Current Braindecode models accept EEG tensors as 3D ``(N, C, T)``.
    The wrapper lets training code stay model-agnostic while avoiding
    version-specific tensor reshaping in the notebook.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ═══════════════════════════════════════════════════════════════
# Model factory
# ═══════════════════════════════════════════════════════════════

def create_model(dataset_name: str, metadata: dict) -> nn.Module:
    """Create the recommended Braindecode model for *dataset_name*.

    The returned model is wrapped with **BraindecodeModelWrapper** so you
    can feed it 3D (N, C, T) tensors directly.
    """
    from braindecode.models import ShallowFBCSPNet, Deep4Net, USleep

    name = dataset_name.upper()
    n_chans   = metadata["n_channels"]
    n_outputs = metadata["num_classes"]
    n_times   = metadata["n_time"]
    sfreq     = metadata["sfreq"]

    def _model_kwargs(model_class, kwargs):
        """Adapt channel/window argument names across Braindecode versions."""
        import inspect

        params = inspect.signature(model_class).parameters
        adapted = dict(kwargs)
        if "n_chans" in adapted and "n_chans" not in params and "in_chans" in params:
            adapted["in_chans"] = adapted.pop("n_chans")
        if (
            "input_window_samples" in adapted
            and "input_window_samples" not in params
            and "n_times" in params
        ):
            adapted["n_times"] = adapted.pop("input_window_samples")
        if "n_classes" in adapted and "n_classes" not in params and "n_outputs" in params:
            adapted["n_outputs"] = adapted.pop("n_classes")
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            adapted = {k: v for k, v in adapted.items() if k in params}
        return adapted

    MODEL_MAP = {
        "SLEEP": (
            USleep,
            {
                "n_chans": n_chans,
                "n_outputs": n_outputs,
                "sfreq": sfreq,
                "input_window_seconds": n_times / sfreq,
                "ensure_odd_conv_size": True,
            },
        ),
        "BCIC2A": (
            ShallowFBCSPNet,
            {
                "n_chans": n_chans,
                "n_outputs": n_outputs,
                "n_times": n_times,
            },
        ),
        "MDD": (
            Deep4Net,
            {
                "n_chans": n_chans,
                "n_outputs": n_outputs,
                "n_times": n_times,
            },
        ),
        "SEED": (
            Deep4Net,
            {
                "n_chans": n_chans,
                "n_outputs": n_outputs,
                "n_times": n_times,
            },
        ),
        "CHINESE": (
            Deep4Net,
            {
                "n_chans": n_chans,
                "n_outputs": n_outputs,
                "n_times": n_times,
            },
        ),
    }

    if name not in MODEL_MAP:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    model_class, kwargs = MODEL_MAP[name]
    raw_model = model_class(**_model_kwargs(model_class, kwargs))
    return BraindecodeModelWrapper(raw_model)


# ═══════════════════════════════════════════════════════════════
# Parameter count helper
# ═══════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
