"""
High-level simulation logging helpers for SimLog.

This module exposes a single main entrypoint:

    log_simulation(sim_type, data, run_id=None, project=None, **hints)

It:
- infers metrics based on sim_type + data type
- logs them into SimLog using your existing tracking backend
- returns the metrics dict
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Type, Optional

# Optional deps – used when available
try:
    import pandas as pd
except ImportError:  # type: ignore
    pd = None

try:
    import numpy as np
except ImportError:  # type: ignore
    np = None

# ---- Hook into your existing SimLog tracking backend -----------------------

# Adjust these imports / function names to match your actual package.
# The expectation is:
#   - create_run(project: str, params: dict) -> str        # returns run_id
#   - log_metrics(run_id: str, metrics: dict) -> None
try:
    from .tracker import create_run, log_metrics  # type: ignore
except Exception as e:
    create_run = None  # type: ignore
    log_metrics = None  # type: ignore
    _BACKEND_IMPORT_ERROR = e
else:
    _BACKEND_IMPORT_ERROR = None


def _ensure_backend():
    if create_run is None or log_metrics is None:
        raise RuntimeError(
            "SimLog simulation helpers could not import the tracking backend. "
            "Please update simlog/simulation.py to import your actual create_run / "
            "log_metrics functions from the correct module. "
            f"Original import error: {_BACKEND_IMPORT_ERROR!r}"
        )


# ---- Handler registry ------------------------------------------------------

_HANDLER_REGISTRY: Dict[Tuple[str, Type[Any]], Callable[[Any, Dict[str, Any]], Dict[str, float]]] = {}


def register_sim_handler(sim_type: str, data_type: Type[Any]):
    """
    Decorator to register a handler for a given (sim_type, data_type).

    A handler has signature:

        handler(data, hints) -> metrics_dict
    """
    def decorator(func: Callable[[Any, Dict[str, Any]], Dict[str, float]]):
        _HANDLER_REGISTRY[(sim_type, data_type)] = func
        return func

    return decorator


def _get_handler(sim_type: str, data: Any) -> Optional[Callable[[Any, Dict[str, Any]], Dict[str, float]]]:
    data_cls = type(data)

    # Exact match first
    key = (sim_type, data_cls)
    if key in _HANDLER_REGISTRY:
        return _HANDLER_REGISTRY[key]

    # Fallback: compatible base classes
    for (st, dt), fn in _HANDLER_REGISTRY.items():
        if st == sim_type and isinstance(data, dt):
            return fn

    return None


# ---- Public API ------------------------------------------------------------

def log_simulation(
    sim_type: str,
    data: Any,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    custom_metrics: Optional[Dict[str, float]] = None,
    **hints: Any,
) -> Dict[str, float]:
    """
    Analyze a simulation result and log key metrics into SimLog.

    Parameters
    ----------
    sim_type:
        High-level kind of simulation, e.g.:
            "time_series", "monte_carlo", "optimization", "agent_based", ...
    data:
        Simulation output. Supported built-ins:
            - pandas.DataFrame (if pandas is installed)
            - numpy.ndarray (if numpy is installed)
        You can register custom handlers for your own types.
    run_id:
        Optional existing SimLog run ID to attach metrics to.
        If None, a new run will be created.
    project:
        Optional project name when creating a new run.
        Ignored if run_id is provided.
    custom_metrics:
        Optional dict of extra metrics to merge into the automatically
        computed metrics. These override any auto-computed keys.
    **hints:
        Extra keyword hints to guide handlers, e.g.:
            time_col="t", value_col="temperature", target="y_true", etc.
        These are passed directly to the handler.

    Returns
    -------
    metrics:
        The final metrics dict that was logged.
    """
    _ensure_backend()

    handler = _get_handler(sim_type, data)

    if handler is None:
        metrics = _generic_metrics(data)
    else:
        metrics = handler(data, hints)

    if custom_metrics:
        metrics.update(custom_metrics)

    # Attach to a run
    if run_id is None:
        # Minimal params: record sim_type + hints as params when creating the run
        params = {
            "sim_type": sim_type,
            **{k: v for k, v in hints.items() if _is_jsonable(v)},
        }
        run_id = create_run(project=project or "default", params=params)  # type: ignore[arg-type]

    # Persist metrics via your backend
    log_metrics(run_id, metrics)  # type: ignore[arg-type]

    return metrics


# ---- Generic fallback metrics ----------------------------------------------

def _generic_metrics(data: Any) -> Dict[str, float]:
    """
    Very lightweight, type-agnostic metrics as a last resort.

    Tries to compute shape/length; if numpy/pandas are present, uses basic stats.
    """
    metrics: Dict[str, float] = {}

    # Length-like
    try:
        length = len(data)  # type: ignore[arg-type]
        metrics["len"] = float(length)
    except Exception:
        pass

    # Numpy array
    if np is not None and isinstance(data, np.ndarray):
        arr = np.asarray(data).astype("float64")
        metrics.update(_basic_array_stats(arr, prefix="data"))
        return metrics

    # Pandas DataFrame
    if pd is not None and isinstance(data, pd.DataFrame):
        return _generic_df_stats(data)

    return metrics


def _basic_array_stats(arr, prefix: str) -> Dict[str, float]:
    """Basic mean/std/min/max for a numeric numpy array."""
    arr = arr.astype("float64")
    if arr.size == 0:
        return {}

    return {
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


def _generic_df_stats(df) -> Dict[str, float]:
    """Simple numeric column stats for a pandas DataFrame."""
    if pd is None:
        return {}

    metrics: Dict[str, float] = {}
    numeric = df.select_dtypes(include="number")

    for col in numeric.columns:
        col_arr = numeric[col].to_numpy(dtype="float64")
        if col_arr.size == 0:
            continue

        metrics[f"{col}_min"] = float(col_arr.min())
        metrics[f"{col}_max"] = float(col_arr.max())
        metrics[f"{col}_mean"] = float(col_arr.mean())
        metrics[f"{col}_std"] = float(col_arr.std())

    metrics["n_rows"] = float(len(df))
    metrics["n_cols"] = float(df.shape[1])

    return metrics


def _is_jsonable(x: Any) -> bool:
    """
    Very cheap check for whether a value is likely to be JSON-serializable
    as a simple param (for run metadata).
    """
    return isinstance(x, (str, int, float, bool, type(None)))


# ---- Built-in handlers -----------------------------------------------------
# You can add more based on your common simulation types.


# 1) Time-series simulations with pandas.DataFrame
if pd is not None:

    @register_sim_handler("time_series", pd.DataFrame)  # type: ignore[misc]
    def _time_series_metrics(df, hints: Dict[str, Any]) -> Dict[str, float]:
        """
        Expected shape: time column + one or more numeric value columns.

        Hints:
            time_col: name of the time column (default: "time")
            value_col: primary value column (default: first numeric col != time_col)
        """
        time_col = hints.get("time_col", "time")
        value_col = hints.get("value_col", None)

        if time_col not in df.columns:
            # Fall back: just take the index as "time"
            t = df.index.to_numpy()
        else:
            t = df[time_col].to_numpy()

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if value_col is None:
            numeric_cols_no_time = [c for c in numeric_cols if c != time_col]
            if not numeric_cols_no_time:
                return {}
            value_col = numeric_cols_no_time[0]

        if value_col not in df.columns:
            return {}

        y = df[value_col].to_numpy()

        import numpy as _np  # local import

        metrics: Dict[str, float] = {
            "ts_min": float(_np.min(y)),
            "ts_max": float(_np.max(y)),
            "ts_mean": float(_np.mean(y)),
            "ts_std": float(_np.std(y)),
        }

        # duration if time is numeric-like
        try:
            duration = float(t[-1] - t[0])
            metrics["ts_duration"] = duration
        except Exception:
            pass

        # crude steady-state heuristic: last 30% of points
        if len(y) > 5:
            start = int(0.7 * len(y))
            last_segment = y[start:]
            metrics["ts_steady_state_mean"] = float(_np.mean(last_segment))
            metrics["ts_steady_state_std"] = float(_np.std(last_segment))

        # final value and delta from start
        metrics["ts_final_value"] = float(y[-1])
        metrics["ts_delta"] = float(y[-1] - y[0])

        return metrics


# 2) Monte Carlo / stochastic sims with numpy.ndarray
if np is not None:

    @register_sim_handler("monte_carlo", np.ndarray)  # type: ignore[misc]
    def _monte_carlo_metrics(arr, hints: Dict[str, Any]) -> Dict[str, float]:
        """
        Expected shape:
            - 1D: scalar outcomes
            - 2D: rows = samples, last column treated as main outcome
        """
        import numpy as _np  # local import

        arr = _np.asarray(arr, dtype="float64")

        if arr.ndim == 1:
            outcomes = arr
        elif arr.ndim >= 2:
            outcomes = arr[:, -1]
        else:
            return {}

        n = outcomes.size
        if n == 0:
            return {}

        mean = float(_np.mean(outcomes))
        std = float(_np.std(outcomes))
        ci_radius = 1.96 * std / (n ** 0.5) if n > 0 else 0.0

        return {
            "mc_n_samples": float(n),
            "mc_mean": mean,
            "mc_std": std,
            "mc_ci95_low": mean - ci_radius,
            "mc_ci95_high": mean + ci_radius,
        }
