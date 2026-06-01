#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import ar1_forecast  # noqa: E402

def main() -> None:
    hist = np.ascontiguousarray(10.0 + np.sin(np.arange(120) * 0.05))
    phi, intercept, steps = 0.7, 0.5, 12
    t0 = time.perf_counter()
    for _ in range(200):
        ar1_forecast(hist, phi, intercept, steps)
    py_s = time.perf_counter() - t0
    try:
        import box_jenkins_for_multivariate_time_series_analysis_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(hist, phi, intercept, steps, 5000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    np.testing.assert_allclose(
        ar1_forecast(hist, phi, intercept, steps),
        np.asarray(rs.ar1_forecast_py(hist, phi, intercept, steps)),
        rtol=1e-12,
    )
    print("Correctness: OK")

if __name__ == "__main__":
    main()
