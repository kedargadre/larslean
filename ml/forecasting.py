"""
ARIMA forecasting module for the ML scoring pipeline.

Provides `forecast_metric` to project a time series forward using ARIMA(1,1,1)
with automatic fallback to linear extrapolation when ARIMA fails or the series
is too short.
"""

import numpy as np
import pandas as pd


def forecast_metric(series: pd.Series, n_periods: int = 6) -> dict:
    """Forecast a metric time series forward by n_periods steps.

    Attempts ARIMA(1,1,1) first; falls back to linear extrapolation on any
    exception or when len(series) < 3.

    Parameters
    ----------
    series : pd.Series
        Monthly values (ideally 12 observations).
    n_periods : int, default 6
        Number of future periods to forecast.

    Returns
    -------
    dict with keys:
        forecast  : np.ndarray  – point forecasts, length n_periods
        lower_ci  : np.ndarray  – 80% CI lower bound
        upper_ci  : np.ndarray  – 80% CI upper bound
        method    : str         – "arima" or "linear"
    """
    if len(series) >= 3:
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(series, order=(1, 1, 1))
            result = model.fit()
            forecast_obj = result.get_forecast(steps=n_periods)
            summary = forecast_obj.summary_frame(alpha=0.2)

            return {
                "forecast": summary["mean"].values,
                "lower_ci": summary["mean_ci_lower"].values,
                "upper_ci": summary["mean_ci_upper"].values,
                "method": "arima",
            }
        except Exception:
            pass  # fall through to linear fallback

    return _linear_fallback(series, n_periods)


def _linear_fallback(series: pd.Series, n_periods: int) -> dict:
    """Linear extrapolation fallback with ±1.5×std confidence interval."""
    n = len(series)
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, series.values.astype(float), deg=1)

    future_x = np.arange(n + 1, n + n_periods + 1, dtype=float)
    forecast = slope * future_x + intercept
    margin = 1.5 * float(np.std(series.values.astype(float)))

    return {
        "forecast": forecast,
        "lower_ci": forecast - margin,
        "upper_ci": forecast + margin,
        "method": "linear",
    }
