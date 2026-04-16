"""Time-Series Forecasting using ARIMA (statsmodels)."""
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def forecast_metric(ts_data: pd.DataFrame, metric: str, periods: int = 6) -> dict:
    """
    Forecast a specific metric forward using ARIMA.

    Args:
        ts_data: DataFrame with 'month' and metric columns
        metric: column name to forecast
        periods: number of months to forecast

    Returns:
        dict with 'dates', 'forecast', 'lower', 'upper', 'historical_dates', 'historical_values'
    """
    if metric not in ts_data.columns or len(ts_data) < 4:
        return _empty_forecast(periods)

    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Prepare time series
        series = ts_data.set_index("month")[metric].dropna()
        if len(series) < 4:
            return _empty_forecast(periods)

        # Fit ARIMA(1,1,1) - simple but effective for short series
        model = ARIMA(series, order=(1, 1, 1))
        fitted = model.fit()

        # Forecast
        forecast_result = fitted.get_forecast(steps=periods)
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=0.2)

        # Generate future dates
        last_date = series.index.max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=periods,
            freq="MS",
        )

        return {
            "dates": future_dates.tolist(),
            "forecast": forecast_mean.values.tolist(),
            "lower": conf_int.iloc[:, 0].values.tolist(),
            "upper": conf_int.iloc[:, 1].values.tolist(),
            "historical_dates": series.index.tolist(),
            "historical_values": series.values.tolist(),
        }

    except Exception as e:
        # Fallback: simple linear extrapolation
        return _linear_fallback(ts_data, metric, periods)


def _linear_fallback(ts_data: pd.DataFrame, metric: str, periods: int) -> dict:
    """Simple linear extrapolation fallback."""
    series = ts_data.set_index("month")[metric].dropna()
    if len(series) < 2:
        return _empty_forecast(periods)

    # Linear trend
    x = np.arange(len(series))
    coeffs = np.polyfit(x, series.values, 1)
    slope, intercept = coeffs

    # Forecast
    future_x = np.arange(len(series), len(series) + periods)
    forecast_vals = slope * future_x + intercept

    # Confidence interval (simple +/- std)
    std = series.std()
    lower = forecast_vals - 1.5 * std
    upper = forecast_vals + 1.5 * std

    last_date = series.index.max()
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=periods,
        freq="MS",
    )

    return {
        "dates": future_dates.tolist(),
        "forecast": forecast_vals.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "historical_dates": series.index.tolist(),
        "historical_values": series.values.tolist(),
    }


def _empty_forecast(periods: int) -> dict:
    """Return empty forecast structure."""
    return {
        "dates": [],
        "forecast": [],
        "lower": [],
        "upper": [],
        "historical_dates": [],
        "historical_values": [],
    }
