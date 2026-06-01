# Box Jenkins for Multivariate Time Series Analysis

Published: 2025-07-15
Medium: [https://medium.com/@kyle-t-jones/box-jenkins-for-multivariate-time-series-analysis-9b7f2ffd2056](https://medium.com/@kyle-t-jones/box-jenkins-for-multivariate-time-series-analysis-9b7f2ffd2056)

## Business context

The Box-Jenkins methodology is a systematic approach to identifying, estimating, and diagnosing time series models. While it is traditionally applied to univariate time series, it can also be extended to multivariate time series using models like Vector Autoregressive (VAR), Vector Moving Average (VMA), and Vector Autoregressive Moving Average (VARMA).

This article applies the methodology to two U.S. economic indicators: the Industrial Production Index (INDPRO) and Retail Sales (RSAFS). Both series are monthly and sourced from the Federal Reserve Economic Data (FRED).

The first step is to test each series for stationarity. Augmented Dickey-Fuller tests indicate that both INDPRO and RSAFS are non-stationary:



## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — AR(1) multi-step forecast. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p box_jenkins_for_multivariate_time_series_analysis_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).