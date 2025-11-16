import json
from pathlib import Path
from typing import Iterable, List, Dict, Any
from .schema import RunRecord

LOG_DIR = Path.home() / ".simlog"
LOG_FILE = LOG_DIR / "runs.jsonl"

def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

def append_run(record: RunRecord) -> None:
    ensure_log_dir()
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")

def load_runs() -> List[Dict[str, Any]]:
    if not LOG_FILE.exists():
        return []
    runs = []
    with LOG_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    return runs
