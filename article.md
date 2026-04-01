# Box-Jenkins for Multivariate Time Series Analysis The Box-Jenkins methodology is a systematic approach to identifying,
estimating, and diagnosing time series models. While it is...

### Box-Jenkins for Multivariate Time Series Analysis
The Box-Jenkins methodology is a systematic approach to identifying,
estimating, and diagnosing time series models. While it is traditionally
applied to univariate time series, it can also be extended to
multivariate time series using models like Vector Autoregressive (VAR),
Vector Moving Average (VMA), and Vector Autoregressive Moving Average
(VARMA).

This article applies the methodology to two U.S. economic indicators:
the Industrial Production Index (INDPRO) and Retail Sales (RSAFS). Both
series are monthly and sourced from the Federal Reserve Economic Data
(FRED).

### Step 1: Identification
The first step is to test each series for stationarity. Augmented
Dickey-Fuller tests indicate that both INDPRO and RSAFS are
non-stationary:

- INDPRO p-value: 0.0822
- RSAFS p-value: 0.9554

Both fail to reject the null hypothesis of a unit root. Differencing is
required.

Log-transformed first differences are then calculated. This ensures
stationarity and makes the two series more directly comparable in scale
and volatility.

### Step 2: Granger Causality
Granger causality tests assess whether one series provides predictive
power for the other. Results show that Retail Sales Granger-causes
Industrial Production at all lags tested (1 through 5), with p-values \<
0.001.

This suggests that changes in consumer spending help forecast changes in
industrial output.

### Step 3: Lag Selection
Using information criteria (AIC, BIC, FPE, HQIC), the optimal lag length
for the VAR model is one month.

### Step 4: Estimation
We fit a VAR(1) model on the differenced series. The regression output
shows:

- Past retail sales positively predict industrial production.
- Past industrial production negatively affects retail sales in the
  short term.

Both effects are statistically significant. The correlation of residuals
is 0.83, suggesting common shocks or omitted shared factors.

### Step 5: Diagnostic Checking
Durbin-Watson statistics for both residual series approach 2.0,
indicating low autocorrelation. Residual plots show no major anomalies.
The model appears well-specified.


### Step 6: Forecasting
Using the fitted VAR(1) model, we forecast both series forward. After
inverting the differencing and log transformation, we plot the
forecasted levels against historical data. The two series continue to
move in tandem, with retail sales leading.

### Summary
This case study demonstrates how to extend the Box-Jenkins framework to
multiple economic time series. The results confirm the intuition that
consumer demand precedes and shapes industrial output.

Multivariate time series models are valuable in macroeconomic
forecasting, business analytics, and financial planning. They allow
analysts to move beyond isolated trends and quantify interdependence
between economic variables.

Use the methods shown here with any pair of correlated economic
indicators to uncover structure, improve predictions, and better
understand the forces shaping your domain.
