# Description: Short example for Box Jenkins for Multivariate Time Series Analysis.


import datetime
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader.data import DataReader
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


start = datetime.datetime(2015, 1, 1)
end = datetime.datetime.today()

# Fetch data from FRED
indpro = DataReader("INDPRO", "fred", start, end)
rsafs = DataReader("RSAFS", "fred", start, end)

# Merge and clean
data = pd.concat([indpro, rsafs], axis=1)
data.columns = ["INDPRO", "RSAFS"]
data = data.dropna()
logger.info(data.head())


# Plot the original data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data["INDPRO"], label="Industrial Production Index")
plt.plot(data.index, data["RSAFS"], label="Retail Sales")
plt.title("Industrial Production vs Retail Sales")
plt.xlabel("Time")
plt.ylabel("Index")
plt.legend()
plt.show()


# Function to test stationarity
def check_stationarity(series, name):
    result = adfuller(series)
    logger.info(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        logger.info(f"{name} is not stationary. Differencing required.")
    else:
        logger.info(f"{name} is stationary.")



def main():
    # Test stationarity for each series
    check_stationarity(data["INDPRO"], "Industrial Production")
    check_stationarity(data["RSAFS"], "Retail Sales")

    # Calculate log returns
    data_diff = np.log(data).diff().dropna()

    # Plot differenced data
    plt.figure(figsize=(12, 6))
    data_diff.plot(title="Differenced Log Returns")
    plt.xlabel("Time")
    plt.ylabel("Log Returns")
    plt.show()

    # Granger Causality Test
    logger.info("\nGranger Causality Tests:")
    logger.info("\nTesting if Retail Sales Granger-causes Industrial Production:")
    gc_test = grangercausalitytests(data_diff[["INDPRO", "RSAFS"]], maxlag=5)

    # Fit VAR model
    model = VAR(data_diff)
    lag_order = model.select_order(maxlags=15)
    logger.info("\nLag Order Selection Criteria:\n", lag_order.summary())

    # Fit model with optimal lags
    fitted_model = model.fit(lag_order.aic)
    logger.info("\nVAR Model Summary:")
    logger.info(fitted_model.summary())

    # Plot residuals
    residuals = fitted_model.resid
    plt.figure(figsize=(12, 6))
    residuals.plot(title="VAR Model Residuals")
    plt.show()

    # Check residual independence
    logger.info("\nDurbin-Watson Test Results:")
    for i, col in enumerate(residuals.columns):
        dw_stat = durbin_watson(residuals[col])
        logger.info(f"Durbin-Watson statistic for {col}: {dw_stat:.2f}")

    # Forecast next 30 days
    forecast = fitted_model.forecast(data_diff.values[-lag_order.aic :], steps=30)
    forecast_index = pd.date_range(start=data.index[-1], periods=30, freq="M")
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)

    # Convert back to index values from returns
    forecast_actual = np.exp(forecast_df.cumsum()) * data.iloc[-1]

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index[-100:],
        data["INDPRO"][-100:],
        label="Industrial Production (Observed)",
        linestyle="-",
    )
    plt.plot(
        data.index[-100:],
        data["RSAFS"][-100:],
        label="Retail Sales (Observed)",
        linestyle="-",
    )
    plt.plot(
        forecast_actual.index,
        forecast_actual["INDPRO"],
        label="Industrial Production (Forecast)",
        linestyle="--",
    )
    plt.plot(
        forecast_actual.index,
        forecast_actual["RSAFS"],
        label="Retail Sales (Forecast)",
        linestyle="--",
    )
    plt.title("VAR Model Forecast")
    plt.xlabel("Time")
    plt.ylabel("Index")
    plt.legend()
    plt.show()


    # Min-max scaling
    scaled = (data - data.min()) / (data.max() - data.min())

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        scaled.index, scaled["INDPRO"], label="Industrial Production (scaled)", linewidth=2
    )
    ax.plot(scaled.index, scaled["RSAFS"], label="Retail Sales (scaled)", linewidth=2)

    # Minimalist styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Scaled Industrial Production and Retail Sales", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Scaled Value (0 to 1)")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
