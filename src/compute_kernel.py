"""AR(1) multi-step forecast."""

from __future__ import annotations

import numpy as np


def ar1_forecast(
    history: np.ndarray, phi: float, intercept: float, steps: int
) -> np.ndarray:
    hist = np.asarray(history, dtype=float)
    out = np.empty(steps, dtype=float)
    y = float(hist[-1]) if len(hist) else 0.0
    for i in range(steps):
        y = intercept + phi * y
        out[i] = y
    return out
