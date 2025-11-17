
# SimLog-Core

SimLog-Core is a lightweight, Python-first logging and metrics framework for simulations, experiments, and model runs.  
It works both as:

- a **decorator** (`@track`) that automatically logs parameters, runtime, metrics, artifacts  
- a **simulation analyzer** (`log_simulation`) that intelligently extracts metrics from many data types  
- a **CLI tool** (`simlog`) for viewing and comparing runs  
- a **JSONL datastore** (`~/.simlog/runs.jsonl`) compatible with notebooks, scripts & cloud workflows  

SimLog is built for scientists, engineers, and ML researchers who want fast, local, zero-setup experiment tracking.

---

## Features

### **1. `@track` — Auto-log Python function runs**
```python
from simlog import track

@track(project="pendulum_sim")
def simulate(theta0, steps):
    ...
    return {
        "metrics": {"energy_loss": 0.12},
        "artifacts": ["plot.png"]
    }
```

This records:

- params (function args)
- metrics (returned from function dict)
- artifacts
- runtime
- git commit
- datetime
- status (success/failed)

All logs are written to:

```
~/.simlog/runs.jsonl
```

---

### **2. `log_simulation` — Intelligent Simulation Metrics**

SimLog can analyze many simulation outputs automatically:

#### Example (time-series)
```python
from simlog import log_simulation
import pandas as pd

df = pd.DataFrame({"t":[0,1,2,3], "temp":[10,12,13,14]})

log_simulation(
    sim_type="time_series",
    data=df,
    project="cooling_tank",
    time_col="t",
    value_col="temp",
)
```

Auto-extracted metrics include:

- min/max/mean/std  
- final value  
- change (`ts_delta`)  
- duration  
- steady-state mean/std  

#### Monte Carlo Example
```python
import numpy as np
log_simulation("monte_carlo", np.random.randn(10000))
```

Outputs mean, std, 95% CI, sample count.

---

### **3. Extend with Your Own Handlers**
```python
from simlog import register_sim_handler

@register_sim_handler("particle_sim", MyParticleArray)
def handle_particles(arr, hints):
    return {
        "avg_energy": arr.energy.mean(),
        "max_speed": arr.speed.max(),
    }
```

---

### **4. CLI Tools**

#### List runs:
```bash
simlog runs
```

#### Show a run:
```bash
simlog show run_abcd1234
```

#### Compare two runs:
```bash
simlog compare run_a run_b
```

---

##  File Format

`~/.simlog/runs.jsonl` contains one JSON object per line, e.g.:

```json
{
  "run_id": "run_abc123",
  "created_at": "2025-02-01T12:30:12Z",
  "user": "vikram",
  "project": "cooling_tank",
  "func_name": "sim:time_series",
  "params": {"sim_type": "time_series"},
  "metrics": {"ts_delta": 4.0},
  "artifacts": [],
  "status": "success",
  "error_message": null,
  "runtime_sec": 0.0,
  "git_commit": "c1a2b3d4"
}
```

---

##  Package Structure

```
simlog/
    __init__.py        # exports track, log_simulation, register_sim_handler
    tracker.py         # implements @track decorator
    simulation.py      # simulation-type intelligent metrics
    storage.py         # JSONL writer/loader
    schema.py          # RunRecord dataclass
    cli.py             # simlog CLI
```

---

## Installation

Install from source:

```bash
pip install -e .
```

Or from PyPI (name: simlog-core):

```bash
pip install simlog-core
```

---

##  Ideal Use Cases

- physics simulations  
- ML hyperparameter sweeps  
- AB testing  
- time-series modeling  
- chemical / biological / fluid simulations  
- Monte Carlo experiments  
- optimization algorithms  
- fusion / tokamak modeling  
- astrophysics (rotation curves, BEC-NFW comparisons, etc.)  

---

##  Vision

SimLog aims to become the simplest experiment-tracking ecosystem:

- **Zero config**
- **Pure Python**
- **Fast local iteration**
- **Notebook friendly**
- **Extendable simulation intelligence**

A massive AI-powered version ("Simora", "Runora", etc.) could eventually add:

- auto-inferred simulation types
- AI suggestions for model improvements
- built-in Jupyter dashboard
- cloud sync + team tracking
- experiment search engine

---

## License
MIT


