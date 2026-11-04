"""Convert solver outputs to NumPy measurements without importing Torch eagerly."""

import numpy as np


def as_numpy(value):
    return (
        value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    )
