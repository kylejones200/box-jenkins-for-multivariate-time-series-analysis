"""Generated from Jupyter notebook: box-jenkins_VAR_multivariate

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR, VARMAX
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series, name):
    result = adfuller(series)
    print(f"{name}: p-value = {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} is not stationary. Differencing required.")
    else:
        print(f"{name} is stationary.")


def simulate_two_interdependent_time_series() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2020-01", periods=100, freq="ME")
    industrial_production = 50 + np.cumsum(np.random.normal(0, 2, 100))
    consumer_price_index = (
        30 + 0.5 * industrial_production + np.random.normal(0, 2, 100)
    )
    data = pd.DataFrame(
        {
            "Industrial_Production": industrial_production,
            "Consumer_Price_Index": consumer_price_index,
        },
        index=time,
    )
    data.plot(figsize=(12, 6), title="Multivariate Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.savefig("Multivariate Time Series")
    plt.show()


def function_to_test_stationarity() -> None:
    check_stationarity(data["Industrial_Production"], "Industrial Production")
    check_stationarity(data["Consumer_Price_Index"], "Consumer Price Index")
    data_diff = data.diff().dropna()
    data_diff.plot(figsize=(12, 6), title="Differenced Multivariate Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
    model = VAR(data_diff)
    lag_order = model.select_order(maxlags=15)
    print("Lag Order Selection Criteria:\n", lag_order.summary())
    fitted_model = model.fit(lag_order.aic)
    print(fitted_model.summary())
    residuals = fitted_model.resid
    residuals.plot(figsize=(12, 6), title="Residuals of VAR Model")
    plt.grid()
    plt.show()
    for i, col in enumerate(residuals.columns):
        dw_stat = durbin_watson(residuals[col])
        print(f"Durbin-Watson statistic for {col}: {dw_stat:.2f}")

    forecast = fitted_model.forecast(data_diff.values[-lag_order.aic :], steps=12)
    forecast_index = pd.date_range(start=data.index[-1], periods=12, freq="M")
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)
    forecast_actual = forecast_df.cumsum() + data.iloc[-1]
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index,
        data["Industrial_Production"],
        label="Industrial Production (Observed)",
        linestyle="-",
    )
    plt.plot(
        data.index,
        data["Consumer_Price_Index"],
        label="Consumer Price Index (Observed)",
        linestyle="-",
    )
    plt.plot(
        forecast_actual.index,
        forecast_actual["Industrial_Production"],
        label="Industrial Production (Forecast)",
        linestyle="--",
    )
    plt.plot(
        forecast_actual.index,
        forecast_actual["Consumer_Price_Index"],
        label="Consumer Price Index (Forecast)",
        linestyle="--",
    )
    plt.title("VAR Model Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("VAR Model Forecast.png")
    plt.show()


def simulate_two_interdependent_time_series_2() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2020-01", periods=100, freq="ME")
    industrial_production = 50 + np.cumsum(np.random.normal(0, 2, 100))
    consumer_price_index = (
        30 + 0.5 * industrial_production + np.random.normal(0, 2, 100)
    )
    data = pd.DataFrame(
        {
            "Industrial_Production": industrial_production,
            "Consumer_Price_Index": consumer_price_index,
        },
        index=time,
    )
    data.plot(figsize=(12, 6), title="Multivariate Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)
    tscv = TimeSeriesSplit(n_splits=5)
    data["day_of_year"] = data.index.dayofyear
    X = data[["day_of_year"]]
    y = data["value"]
    predictions = []
    holdouts = []
    for fold_idx, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = (X.iloc[train_index], X.iloc[test_index])
        y_train, y_test = (y.iloc[train_index], y.iloc[test_index])
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(pd.Series(y_pred, index=y_test.index))
        holdouts.append(y_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold_idx + 1}: MSE = {mse:.4f}")

    all_predictions = pd.concat(predictions).sort_index()
    all_holdouts = pd.concat(holdouts).sort_index()
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index,
        data["value"],
        label="Actual Values (Full)",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        all_holdouts.index,
        all_holdouts,
        label="Hold-Out Data",
        color="green",
        linewidth=2,
    )
    plt.plot(
        all_predictions.index,
        all_predictions,
        label="Forecasted Values",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.title("Time Series Forecasting with TimeSeriesSplit")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_series_split_forecast.png")
    plt.show()


def create_a_sample_time_series_dataframe() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    # Create a sample time series dataframe
    data = pd.DataFrame(
        {
            "date": pd.date_range(start="1/1/2020", periods=100, freq="D"),
            "value": np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100),
        }
    )
    # Ensure 'date' is a datetime object
    data["date"] = pd.to_datetime(data["date"])
    # Set 'date' as the index for plotting
    data.set_index("date", inplace=True)
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    # Prepare features and target
    data["day_of_year"] = data.index.dayofyear  # Example feature
    X = data[["day_of_year"]]
    y = data["value"]
    # Initialize storage for fold 5
    fold_5
    _predictions = None
    fold_5
    _holdouts = None
    # Perform cross-validation
    for fold_idx, (train_index, test_index) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # Train a model
        model = LinearRegression()
        model.fit(X_train, y_train)
        # Predict on the test set
        y_pred = model.predict(X_test)
        # Store predictions and hold-outs for fold 5
        if fold_idx == 5:
            fold_5_predictions = pd.Series(y_pred, index=y_test.index)
            fold_5_holdouts = y_test

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold {fold_idx}: MSE = {mse:.4f}")

    # Plot the actual values, hold-outs, and predictions for fold 5
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index,
        data["value"],
        label="Actual Values (Full)",
        color="blue",
        linewidth=2,
    )
    if fold_5_holdouts is not None:
        plt.plot(
            fold_5_holdouts.index,
            fold_5_holdouts,
            label="Hold-Out Data (Fold 5)",
            color="green",
            linewidth=2,
        )
    if fold_5_predictions is not None:
        plt.plot(
            fold_5_predictions.index,
            fold_5_predictions,
            label="Forecasted Values (Fold 5)",
            color="red",
            linestyle="--",
            linewidth=2,
        )
    plt.title("Time Series Forecasting with TimeSeriesSplit (Fold 5)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("time_series_split_fold_5_forecast.png")
    plt.show()


def simulate_two_interdependent_time_series_3() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2020-01", periods=100, freq="ME")
    industrial_production = 50 + np.cumsum(np.random.normal(0, 2, 100))
    consumer_price_index = (
        30 + 0.5 * industrial_production + np.random.normal(0, 2, 100)
    )
    data = pd.DataFrame(
        {
            "Industrial_Production": industrial_production,
            "Consumer_Price_Index": consumer_price_index,
        },
        index=time,
    )
    data.plot(figsize=(12, 6), title="Multivariate Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
    check_stationarity(data["Industrial_Production"], "Industrial Production")
    check_stationarity(data["Consumer_Price_Index"], "Consumer Price Index")
    data_diff = data.diff().dropna()
    train_size = int(len(data_diff) * 0.8)
    train_data = data_diff.iloc[:train_size]
    test_data = data_diff.iloc[train_size:]
    model = VAR(train_data)
    lag_order = model.select_order(maxlags=15)
    print("Lag Order Selection Criteria:\n", lag_order.summary())
    optimal_lag = lag_order.aic
    fitted_model = model.fit(optimal_lag)
    print(fitted_model.summary())
    forecast = fitted_model.forecast(
        train_data.values[-optimal_lag:], steps=len(test_data)
    )
    forecast_index = test_data.index
    forecast_df = pd.DataFrame(
        forecast, index=forecast_index, columns=data_diff.columns
    )
    forecast_actual = forecast_df.cumsum() + data.iloc[train_size - 1]
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.iloc[:train_size].index,
        data.iloc[:train_size]["Industrial_Production"],
        label="Train Data (Industrial Production)",
        color="blue",
    )
    plt.plot(
        data.iloc[train_size:].index,
        data.iloc[train_size:]["Industrial_Production"],
        label="Test Data (Industrial Production)",
        color="green",
    )
    plt.plot(
        forecast_actual.index,
        forecast_actual["Industrial_Production"],
        label="Forecasted Data (Industrial Production)",
        linestyle="--",
        color="red",
    )
    plt.title("VAR Model Forecast with Train, Test, and Predictions")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("train_test_forecast_plot.png")
    plt.show()
    residuals = fitted_model.resid
    residuals.plot(figsize=(12, 6), title="Residuals of VAR Model")
    plt.grid()
    plt.show()
    for i, col in enumerate(residuals.columns):
        dw_stat = durbin_watson(residuals[col])
        print(f"Durbin-Watson statistic for {col}: {dw_stat:.2f}")


def simulate_two_interdependent_time_series_4() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2020-01", periods=100, freq="ME")
    industrial_production = 50 + np.cumsum(np.random.normal(0, 2, 100))
    consumer_price_index = (
        30 + 0.5 * industrial_production + np.random.normal(0, 2, 100)
    )
    data = pd.DataFrame(
        {
            "Industrial_Production": industrial_production,
            "Consumer_Price_Index": consumer_price_index,
        },
        index=time,
    )
    data.plot(figsize=(12, 6), title="Multivariate Time Series")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
    check_stationarity(data["Industrial_Production"], "Industrial Production")
    check_stationarity(data["Consumer_Price_Index"], "Consumer Price Index")
    data_diff = data.diff().dropna()
    train_size = int(len(data_diff) * 0.8)
    train_data = data_diff.iloc[:train_size]
    test_data = data_diff.iloc[train_size:]
    var_model = VAR(train_data)
    var_lag_order = var_model.select_order(maxlags=15)
    print("VAR Lag Order Selection Criteria:\n", var_lag_order.summary())
    var_optimal_lag = var_lag_order.aic
    var_fitted_model = var_model.fit(var_optimal_lag)
    print(var_fitted_model.summary())
    varma_model = VARMAX(train_data, order=(var_optimal_lag, 1))
    varma_fitted_model = varma_model.fit(disp=False)
    print("VARMA Model Summary:\n", varma_fitted_model.summary())
    var_forecast = var_fitted_model.forecast(
        train_data.values[-var_optimal_lag:], steps=len(test_data)
    )
    varma_forecast = varma_fitted_model.forecast(steps=len(test_data))
    forecast_index = test_data.index
    var_forecast_df = pd.DataFrame(
        var_forecast, index=forecast_index, columns=data_diff.columns
    )
    varma_forecast_df = pd.DataFrame(
        varma_forecast, index=forecast_index, columns=data_diff.columns
    )
    var_forecast_actual = var_forecast_df.cumsum() + data.iloc[train_size - 1]
    varma_forecast_actual = varma_forecast_df.cumsum() + data.iloc[train_size - 1]
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.iloc[:train_size].index,
        data.iloc[:train_size]["Industrial_Production"],
        label="Train Data",
        color="blue",
    )
    plt.plot(
        data.iloc[train_size:].index,
        data.iloc[train_size:]["Industrial_Production"],
        label="Test Data",
        color="green",
    )
    plt.plot(
        var_forecast_actual.index,
        var_forecast_actual["Industrial_Production"],
        label="Forecasted Data (VAR)",
        linestyle="--",
        color="red",
    )
    plt.title("VAR Model Forecast with Train, Test, and Predictions")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.iloc[:train_size].index,
        data.iloc[:train_size]["Industrial_Production"],
        label="Train Data",
        color="blue",
    )
    plt.plot(
        data.iloc[train_size:].index,
        data.iloc[train_size:]["Industrial_Production"],
        label="Test Data",
        color="green",
    )
    plt.plot(
        varma_forecast_actual.index,
        varma_forecast_actual["Industrial_Production"],
        label="Forecasted Data (VARMA)",
        linestyle="--",
        color="purple",
    )
    plt.title("VARMA Model Forecast with Train, Test, and Predictions")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    var_residuals = var_fitted_model.resid
    var_residuals.plot(figsize=(12, 6), title="Residuals of VAR Model")
    plt.grid()
    plt.show()
    varma_residuals = varma_fitted_model.resid
    varma_residuals.plot(figsize=(12, 6), title="Residuals of VARMA Model")
    plt.grid()
    plt.show()
    print("Durbin-Watson Statistics:")
    for i, col in enumerate(var_residuals.columns):
        dw_stat_var = durbin_watson(var_residuals[col])
        print(f"VAR - {col}: {dw_stat_var:.2f}")

    for i, col in enumerate(varma_residuals.columns):
        dw_stat_varma = durbin_watson(varma_residuals[col])
        print(f"VARMA - {col}: {dw_stat_varma:.2f}")


def scale_the_data() -> None:
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(data), index=data.index, columns=data.columns
    )
    data_diff_scaled = data_scaled.diff().dropna()
    varma_model = VARMAX(data_diff_scaled, order=(2, 1))
    varma_fitted_model = varma_model.fit(maxiter=1000, disp=False)
    print(varma_fitted_model.summary())


def simulate_two_interdependent_time_series_5() -> None:
    np.random.seed(42)
    time = pd.date_range(start="2020-01", periods=100, freq="ME")
    industrial_production = 50 + np.cumsum(np.random.normal(0, 2, 100))
    data = pd.DataFrame({"Industrial_Production": industrial_production}, index=time)
    data.plot(figsize=(12, 6), title="Time Series Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()
    result = adfuller(data)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    if result[1] > 0.05:
        print("The series is non-stationary. Differencing required.")

    data_diff = data.diff().dropna()
    data_diff.plot(figsize=(12, 6), title="Differenced Time Series")
    plt.xlabel("Time")
    plt.ylabel("Differenced Value")
    plt.grid()
    plt.show()
    plot_acf(data_diff, lags=20, title="ACF of Differenced Series")
    plt.show()
    plot_pacf(data_diff, lags=20, title="PACF of Differenced Series")
    plt.show()
    model = auto_arima(
        data,
        seasonal=False,
        trace=True,
        stepwise=True,
        suppress_warnings=True,
        max_p=5,
        max_q=5,
        max_d=2,
    )
    print(model.summary())
    model.fit(data)
    forecast, conf_int = model.predict(n_periods=12, return_conf_int=True)
    forecast_index = pd.date_range(
        data.index[-1] + pd.DateOffset(1), periods=12, freq="M"
    )
    forecast_series = pd.Series(forecast, index=forecast_index)
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Actual Data", color="blue")
    plt.plot(forecast_series, label="Forecast", color="red", linestyle="--")
    plt.fill_between(
        forecast_index,
        conf_int[:, 0],
        conf_int[:, 1],
        color="pink",
        alpha=0.3,
        label="Confidence Interval",
    )
    plt.title("ARIMA Model Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main() -> None:
    simulate_two_interdependent_time_series()
    function_to_test_stationarity()
    simulate_two_interdependent_time_series_2()
    create_a_sample_time_series_dataframe()
    simulate_two_interdependent_time_series_3()
    simulate_two_interdependent_time_series_4()
    scale_the_data()
    simulate_two_interdependent_time_series_5()


if __name__ == "__main__":
    main()
