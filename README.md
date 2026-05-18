# Box Jenkins for Multivariate Time Series Analysis

Published: 2025-07-15
Medium: [https://medium.com/@kyle-t-jones/box-jenkins-for-multivariate-time-series-analysis-9b7f2ffd2056](https://medium.com/@kyle-t-jones/box-jenkins-for-multivariate-time-series-analysis-9b7f2ffd2056)

## Business context

The Box-Jenkins methodology is a systematic approach to identifying, estimating, and diagnosing time series models. While it is traditionally applied to univariate time series, it can also be extended to multivariate time series using models like Vector Autoregressive (VAR), Vector Moving Average (VMA), and Vector Autoregressive Moving Average (VARMA).

This article applies the methodology to two U.S. economic indicators: the Industrial Production Index (INDPRO) and Retail Sales (RSAFS). Both series are monthly and sourced from the Federal Reserve Economic Data (FRED).

The first step is to test each series for stationarity. Augmented Dickey-Fuller tests indicate that both INDPRO and RSAFS are non-stationary:

## About

Place the code for this article in this repository.
The original article export is saved as `article.md`.

## Files

Add your `.ipynb`, `.py`, `.yaml`, `.js`, `.ts`, or other project files here.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).