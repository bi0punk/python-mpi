from pydantic import BaseModel
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class AlgorithmType(str, Enum):
    primes = "primes"
    pi_monte_carlo = "pi_monte_carlo"
    stress = "stress"


class TaskCreate(BaseModel):
    algorithm: AlgorithmType
    params: dict
    nodes: int = 1


class TaskResult(BaseModel):
    task_id: int
    result: dict
    elapsed: float
    worker_count: int


class Task(BaseModel):
    id: int
    algorithm: str
    params: dict
    nodes: int
    status: TaskStatus
    result: Optional[dict] = None
    elapsed: Optional[float] = None
    created_at: str
    completed_at: Optional[str] = None
