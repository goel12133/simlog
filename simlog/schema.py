from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid

@dataclass
class RunRecord:
    run_id: str
    created_at: str
    user: str
    project: str

    func_name: str
    params: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: List[str]

    status: str
    error_message: Optional[str]
    runtime_sec: float

    git_commit: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def new(
        user: str,
        project: str,
        func_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[List[str]],
        status: str,
        error_message: Optional[str],
        runtime_sec: float,
        git_commit: Optional[str],
    ) -> "RunRecord":
        return RunRecord(
            run_id=f"run_{uuid.uuid4().hex}",
            created_at=datetime.utcnow().isoformat() + "Z",
            user=user,
            project=project,
            func_name=func_name,
            params=params,
            metrics=metrics or {},
            artifacts=artifacts or [],
            status=status,
            error_message=error_message,
            runtime_sec=runtime_sec,
            git_commit=git_commit,
        )
