# simlog/__init__.py

from .tracker import track
from .storage import load_runs
from . import simulation as _simulation

# Re-export from simulation module
log_simulation = _simulation.log_simulation
register_sim_handler = _simulation.register_sim_handler

__all__ = [
    "track",
    "log_simulation",
    "register_sim_handler",
    "load_runs",
]
