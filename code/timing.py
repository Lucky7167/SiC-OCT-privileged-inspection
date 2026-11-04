from __future__ import annotations

import numpy as np
import pandas as pd

PIPELINE_STAGES = {
    "microscopy",
    "robot_motion",
    "oct_acquisition",
    "oct_reconstruction",
}


def adjudication_durations(events):
    frame = pd.DataFrame(events).copy()
    required = {"display_time", "decision_time"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"adjudication events are missing {sorted(missing)}")
    display = frame["display_time"]
    decision = frame["decision_time"]
    if not pd.api.types.is_numeric_dtype(display) or not pd.api.types.is_numeric_dtype(decision):
        display = pd.to_datetime(display)
        decision = pd.to_datetime(decision)
        durations = (decision - display).dt.total_seconds()
    else:
        durations = decision.astype(float) - display.astype(float)
    if durations.isna().any() or not np.isfinite(durations).all():
        raise ValueError("adjudication event times must be finite")
    if (durations < 0).any():
        raise ValueError("decision_time cannot precede display_time")
    frame["adjudication_seconds"] = durations.astype(float)
    return frame


def summarize_adjudication(events):
    durations = adjudication_durations(events)
    group_columns = [column for column in ("wafer", "reader") if column in durations]
    if not group_columns:
        return pd.DataFrame([{
            "n_volumes": len(durations),
            "total_seconds": durations.adjudication_seconds.sum(),
            "mean_seconds": durations.adjudication_seconds.mean(),
            "median_seconds": durations.adjudication_seconds.median(),
        }])
    result = durations.groupby(group_columns, as_index=False).adjudication_seconds.agg(
        ["size", "sum", "mean", "median"]
    ).reset_index()
    return result.rename(columns={
        "size": "n_volumes",
        "sum": "total_seconds",
        "mean": "mean_seconds",
        "median": "median_seconds",
    })


def _numeric_acquisition_time(microscopy, robot, acquisition, reconstruction):
    values = np.asarray([microscopy, robot, acquisition, reconstruction], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("pipeline durations must be finite and non-negative")
    return float(values[0] + values[1] + max(values[2], values[3]))


def acquisition_pipeline_time(
    events,
    robot_motion_seconds=None,
    oct_acquisition_seconds=None,
    oct_reconstruction_seconds=None,
):
    if np.isscalar(events):
        durations = (robot_motion_seconds, oct_acquisition_seconds, oct_reconstruction_seconds)
        if any(value is None for value in durations):
            raise ValueError("numeric pipeline timing requires all four stage durations")
        return _numeric_acquisition_time(events, *durations)
    frame = pd.DataFrame(events).copy()
    required = {"stage", "start", "end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pipeline events are missing {sorted(missing)}")
    frame = frame[frame.stage.isin(PIPELINE_STAGES)]
    if frame.empty:
        return 0.0
    start = frame["start"]
    end = frame["end"]
    if not pd.api.types.is_numeric_dtype(start) or not pd.api.types.is_numeric_dtype(end):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        if (end < start).any():
            raise ValueError("pipeline event end cannot precede its start")
        origin = start.min()
        return float((end.max() - origin).total_seconds())
    if start.isna().any() or end.isna().any():
        raise ValueError("pipeline event times must be finite")
    if not np.isfinite(start.to_numpy(float)).all() or not np.isfinite(end.to_numpy(float)).all():
        raise ValueError("pipeline event times must be finite")
    if (end < start).any():
        raise ValueError("pipeline event end cannot precede its start")
    return float(end.max() - start.min())


def time_to_result(acquisition_seconds, adjudication_seconds):
    acquisition = float(acquisition_seconds)
    adjudication = float(adjudication_seconds)
    if not np.isfinite([acquisition, adjudication]).all():
        raise ValueError("time-to-result components must be finite")
    if acquisition < 0 or adjudication < 0:
        raise ValueError("time-to-result components must be non-negative")
    return acquisition + adjudication
