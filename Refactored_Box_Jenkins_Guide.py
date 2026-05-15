"""Generated from Jupyter notebook: Box-Jenkins Methodology with timesmith and plotsmith

Magics and shell lines are commented out. Run with a normal Python interpreter."""


# --- code cell ---

from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotsmith as ps
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR


def main():
    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")


    # --- code cell ---

    # Load economic data from FRED using pandas_datareader
    start = datetime(2015, 1, 1)
    end = datetime.now()

    try:
        indpro_raw = web.DataReader("INDPRO", "fred", start, end)
        rsafs_raw = web.DataReader("RSAFS", "fred", start, end)

        # Extract columns (FRED returns lowercase column names)
        indpro_col = indpro_raw.columns[0]
        rsafs_col = rsafs_raw.columns[0]

        indpro = indpro_raw[indpro_col]
        rsafs = rsafs_raw[rsafs_col]

        # Merge and clean
        data_var = pd.DataFrame(
            {"Industrial_Production": indpro, "Retail_Sales": rsafs}
        ).dropna()

        print(f"✓ Loaded multivariate data: {len(data_var)} observations")
        print(f"  Variables: {list(data_var.columns)}")
        print(f"\nDate range: {data_var.index.min()} to {data_var.index.max()}")
        print(data_var.head())

    except Exception as e:
        print(f"Could not fetch FRED data: {e}")
        print("Creating synthetic multivariate data instead...")

        # Create synthetic data
        np.random.seed(42)
        time = pd.date_range(start="2015-01", periods=100, freq="ME")
        indpro = 50 + np.cumsum(np.random.normal(0, 2, 100))
        rsafs = 30 + 0.5 * indpro + np.random.normal(0, 2, 100)

        data_var = pd.DataFrame(
            {"Industrial_Production": indpro, "Retail_Sales": rsafs}, index=time
        )

        print(f"✓ Created synthetic data: {len(data_var)} observations")


    # --- code cell ---

    # Visualize multivariate data using plotsmith
    ps.plot_timeseries(
        data_var,
        title="Multivariate Time Series",
        xlabel="Time",
        ylabel="Value",
        figsize=(15, 6),
    )


    # --- code cell ---

    # Check stationarity for all series
    print("=" * 70)
    print("STATIONARITY TEST FOR MULTIVARIATE SERIES")
    print("=" * 70 + "\n")

    for col in data_var.columns:
        result = adfuller(data_var[col].dropna())
        print(f"{col}:")
        print(f"  p-value = {result[1]:.4f}", end="")

        if result[1] > 0.05:
            print(" → Non-stationary. Differencing required.")
        else:
            print(" → Stationary.")
        print()

    # Apply differencing using pandas (timesmith extends pandas operations)
    data_var_diff = data_var.diff().dropna()

    print("\nAfter differencing:")
    for col in data_var_diff.columns:
        result = adfuller(data_var_diff[col].dropna())
        print(f"{col}: p-value = {result[1]:.4f}")

    # Visualize differenced data using plotsmith
    ps.plot_timeseries(
        data_var_diff,
        title="Differenced Multivariate Series",
        xlabel="Time",
        ylabel="Differenced Value",
        figsize=(15, 6),
    )


    # --- code cell ---

    # Test Granger causality
    print("=" * 70)
    print("GRANGER CAUSALITY TEST")
    print("=" * 70)
    print("\nTests if one series helps predict another.")
    print("Null hypothesis: X does NOT Granger-cause Y\n")

    col1, col2 = data_var_diff.columns[0], data_var_diff.columns[1]
    print(f"\nTesting if {col2} Granger-causes {col1}:")
    print("=" * 70)

    gc_result = grangercausalitytests(data_var_diff[[col1, col2]], maxlag=5, verbose=False)

    # Extract p-values
    print("\nSummary of p-values (F-test):")
    for lag in range(1, 6):
        p_value = gc_result[lag][0]["ssr_ftest"][1]
        print(f"  Lag {lag}: p-value = {p_value:.4f}", end="")
        if p_value < 0.05:
            print(" → Significant Granger causality")
        else:
            print(" → No significant Granger causality")


    # --- code cell ---

    # Create and fit VAR model
    print("=" * 70)
    print("VAR MODEL ESTIMATION")
    print("=" * 70 + "\n")

    # Create VAR model
    model_var = VAR(data_var_diff)

    # Select optimal lag order
    lag_order = model_var.select_order(maxlags=15)
    print("Lag Order Selection Criteria:")
    print(lag_order.summary())

    # Fit model with optimal lags (using AIC)
    fitted_var = model_var.fit(lag_order.aic)
    print(f"\n✓ Fitted VAR({lag_order.aic}) model")
    print(f"\nModel Summary:")
    print(fitted_var.summary())


    # --- code cell ---

    # Get residuals and visualize using plotsmith
    print("=" * 70)
    print("VAR MODEL DIAGNOSTICS")
    print("=" * 70 + "\n")

    residuals_var = fitted_var.resid

    # Visualize residuals using plotsmith
    for col in residuals_var.columns:
        ps.plot_timeseries(
            residuals_var[col],
            title=f"Residuals: {col}",
            xlabel="Time",
            ylabel="Residual",
            figsize=(12, 4),
        )

    # Durbin-Watson test for serial correlation
    print("\nDurbin-Watson Test (should be close to 2):")
    for col in residuals_var.columns:
        dw_stat = durbin_watson(residuals_var[col].values)
        print(f"  {col}: {dw_stat:.2f}")


    # --- code cell ---

    # Forecast next 20 periods
    print("=" * 70)
    print("VAR FORECASTING")
    print("=" * 70 + "\n")

    n_forecast = 20
    forecast_var = fitted_var.forecast(
        data_var_diff.values[-lag_order.aic :], steps=n_forecast
    )

    # Create forecast DataFrame
    forecast_index = pd.date_range(
        start=data_var.index[-1], periods=n_forecast + 1, freq=data_var.index.freq or "ME"
    )[1:]

    forecast_var_df = pd.DataFrame(
        forecast_var, index=forecast_index, columns=data_var.columns
    )

    # Convert back from differences
    forecast_actual = forecast_var_df.cumsum() + data_var.iloc[-1]

    print("Forecast (first 10 periods):")
    print(forecast_actual.head(10))

    # Visualize forecasts using plotsmith
    for col in data_var.columns:
        historical_subset = data_var[col].iloc[-100:]
        forecast_subset = forecast_actual[col]

        ps.plot_model_comparison(
            data=historical_subset,
            models={"VAR Forecast": forecast_subset},
            title=f"VAR Forecast: {col}",
            xlabel="Time",
            ylabel="Value",
            figsize=(12, 4),
        )


if __name__ == "__main__":
    main()
