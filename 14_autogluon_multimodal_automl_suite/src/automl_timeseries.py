"""Thin wrapper around AutoGluon's TimeSeriesPredictor -- a distinct AutoGluon
subpackage (autogluon.timeseries) from the tabular predictor used elsewhere in
this suite, demonstrating multi-series probabilistic forecasting.
"""
from __future__ import annotations

from pathlib import Path

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def to_ts_dataframe(df, id_column: str, timestamp_column: str):
    return TimeSeriesDataFrame.from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)


def fit_forecaster(train_ts, target: str, prediction_length: int, save_path: Path, time_limit: int, eval_metric: str):
    predictor = TimeSeriesPredictor(
        target=target,
        prediction_length=prediction_length,
        path=str(save_path),
        eval_metric=eval_metric,
        verbosity=0,
    ).fit(
        train_ts,
        presets="medium_quality",
        time_limit=time_limit,
    )
    return predictor
