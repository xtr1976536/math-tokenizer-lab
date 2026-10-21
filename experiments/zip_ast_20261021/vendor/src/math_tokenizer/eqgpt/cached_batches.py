"""Cache exact author-collated batches while preserving DataLoader RNG draws."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader


def identity(value):
    return value


def cache_batches(loader, device):
    """Same batch order, padding, values and epoch iterator RNG as the source.

    Preparation's temporary iterator consumes one CPU base-seed draw; restore it.
    The returned DataLoader consumes the usual draw on every training epoch.
    Original author train_step still handles the loss, gradients and optimizer.
    CUDA tensors stay resident; its unchanged .to(device) becomes a no-op.
    """
    rng = torch.get_rng_state()
    try:
        values = tuple((x.to(device), y.to(device)) for x, y in loader)
    finally:
        torch.set_rng_state(rng)
    return DataLoader(values, batch_size=None, collate_fn=identity, shuffle=False)
