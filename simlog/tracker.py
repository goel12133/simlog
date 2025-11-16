import functools
import inspect
import os
import subprocess
import time
from typing import Callable, Any, Dict

from .schema import RunRecord
from .storage import append_run

def _get_git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None

def _extract_params(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)

def track(fn: Callable | None = None, *, project: str = "default") -> Callable:
    """
    Usage:
        @track
        def run_sim(...):
            ...

        @track(project="bec_vs_nfw")
        def run_sim(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user = os.getenv("SIMLOG_USER", os.getenv("USER", "unknown"))
            params = _extract_params(func, *args, **kwargs)
            git_commit = _get_git_commit()

            start = time.time()
            status = "success"
            error_message = None
            metrics: Dict[str, float] = {}
            artifacts = []

            try:
                result = func(*args, **kwargs)
            except Exception as e:
                status = "failed"
                error_message = str(e)
                runtime = time.time() - start
                record = RunRecord.new(
                    user=user,
                    project=project,
                    func_name=func.__name__,
                    params=params,
                    metrics=metrics,
                    artifacts=artifacts,
                    status=status,
                    error_message=error_message,
                    runtime_sec=runtime,
                    git_commit=git_commit,
                )
                append_run(record)
                raise

            runtime = time.time() - start

            # convention: function returns dict with optional "metrics" and "artifacts"
            if isinstance(result, dict):
                metrics = result.get("metrics", {}) or {}
                artifacts = result.get("artifacts", []) or []

            record = RunRecord.new(
                user=user,
                project=project,
                func_name=func.__name__,
                params=params,
                metrics=metrics,
                artifacts=artifacts,
                status=status,
                error_message=error_message,
                runtime_sec=runtime,
                git_commit=git_commit,
            )
            append_run(record)
            return result
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator
