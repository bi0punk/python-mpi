import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "data" / "mpi_jobs.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            params TEXT NOT NULL,
            nodes INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'pending',
            result TEXT,
            elapsed REAL,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS workers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rank INTEGER NOT NULL UNIQUE,
            host TEXT NOT NULL,
            last_ping TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


def create_task(algorithm: str, params: dict, nodes: int) -> int:
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO tasks (algorithm, params, nodes, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
        (algorithm, json.dumps(params), nodes, now),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()
    return task_id


def get_next_pending_task():
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM tasks WHERE status = 'pending' ORDER BY id ASC LIMIT 1"
    ).fetchone()
    if row:
        task_id = row["id"]
        conn.execute("UPDATE tasks SET status = 'running' WHERE id = ?", (task_id,))
        conn.commit()
        task = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        conn.close()
        return dict(task)
    conn.close()
    return None


def complete_task(task_id: int, result: dict, elapsed: float):
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE tasks SET status = 'completed', result = ?, elapsed = ?, completed_at = ? WHERE id = ?",
        (json.dumps(result), elapsed, now, task_id),
    )
    conn.commit()
    conn.close()


def fail_task(task_id: int, error: str):
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE tasks SET status = 'failed', result = ?, completed_at = ? WHERE id = ?",
        (json.dumps({"error": error}), now, task_id),
    )
    conn.commit()
    conn.close()


def get_all_tasks():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_task(task_id: int) -> Optional[dict]:
    conn = get_connection()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def register_worker(rank: int, host: str):
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO workers (rank, host, last_ping) VALUES (?, ?, ?)",
        (rank, host, now),
    )
    conn.commit()
    conn.close()


def get_workers():
    conn = get_connection()
    rows = conn.execute("SELECT * FROM workers ORDER BY rank").fetchall()
    conn.close()
    return [dict(r) for r in rows]
