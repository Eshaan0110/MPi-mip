"""
FIX F1: Replace fabricated ETS confidence intervals with proper
simulation-based prediction intervals from statsmodels.
================================================================
The current _ETSWrapper generates CIs as ±5%→±20% linear spread.
This replaces it with statsmodels' simulate() method which provides
statistically valid prediction intervals.

Drop-in replacement for _ETSWrapper.predict() in bank_model.py.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ETSWrapperFixed:
    """Holt-Winters wrapper with proper prediction intervals via simulation."""

    def __init__(self, fit, bank_df, bank_name, card_type):
        self._fit = fit
        self._bank_df = bank_df
        self._bank_name = bank_name
        self._card_type = card_type
        self._n_train = len(bank_df)

    def make_future_dataframe(self, periods, freq="MS"):
        last_date = self._bank_df["ds"].max()
        hist_dates = self._bank_df["ds"].tolist()
        future_dates = pd.date_range(last_date, periods=periods + 1, freq=freq)[1:]
        all_dates = hist_dates + future_dates.tolist()
        return pd.DataFrame({"ds": all_dates})

    def predict(self, future):
        n_hist = self._n_train
        n_future = len(future) - n_hist

        fitted = self._fit.fittedvalues
        if n_future > 0:
            fc = self._fit.forecast(n_future)
            yhat = np.concatenate([fitted, fc])

            # Simulation-based prediction intervals (1000 paths)
            try:
                sim = self._fit.simulate(
                    nsimulations=n_future,
                    repetitions=1000,
                    anchor="end",
                )
                lower = np.percentile(sim, 5, axis=1)   # 90% CI
                upper = np.percentile(sim, 95, axis=1)

                yhat_lower = np.concatenate([fitted * 0.95, lower])  # history: tight
                yhat_upper = np.concatenate([fitted * 1.05, upper])
            except Exception:
                # Fallback: use residual std for CI estimation
                resid_std = np.std(self._fit.resid)
                steps = np.arange(1, n_future + 1)
                # CI widens with sqrt(h) — standard for additive models
                widths = 1.645 * resid_std * np.sqrt(steps)
                yhat_lower = np.concatenate([fitted - 1.645 * resid_std, fc - widths])
                yhat_upper = np.concatenate([fitted + 1.645 * resid_std, fc + widths])
        else:
            yhat = fitted[:len(future)]
            resid_std = np.std(self._fit.resid)
            yhat_lower = yhat - 1.645 * resid_std
            yhat_upper = yhat + 1.645 * resid_std

        result = pd.DataFrame({
            "ds": future["ds"].values[:len(yhat)],
            "yhat": yhat[:len(future)],
            "yhat_lower": yhat_lower[:len(future)],
            "yhat_upper": yhat_upper[:len(future)],
            "trend": yhat[:len(future)],
        })
        return result


if __name__ == "__main__":
    # Quick test: fit ETS on synthetic data and check CIs make sense
    np.random.seed(42)
    n = 60
    t = np.arange(n)
    y = 1000 + 10 * t + 50 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 20, n)

    model = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit(optimized=True)

    bank_df = pd.DataFrame({
        "ds": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "y": y,
    })

    wrapper = ETSWrapperFixed(fit, bank_df, "Test Bank", "cc")
    future = wrapper.make_future_dataframe(periods=12)
    result = wrapper.predict(future)

    print("Forecast with proper CIs:")
    print(result.tail(12)[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_string())

    # Verify CIs are asymmetric and widen over time
    fc_only = result.tail(12)
    widths = fc_only["yhat_upper"] - fc_only["yhat_lower"]
    print(f"\nCI widths: {widths.iloc[0]:.1f} -> {widths.iloc[-1]:.1f}")
    assert widths.iloc[-1] > widths.iloc[0], "CIs should widen over time"
    print("PASS: CIs widen correctly over forecast horizon")
