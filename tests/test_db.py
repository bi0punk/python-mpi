import json
import pytest
from master import db


@pytest.fixture(autouse=True)
def setup_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", db_path)
    db.init_db()
    yield
    if db_path.exists():
        db_path.unlink()


class TestDatabase:
    def test_init_db_creates_tables(self):
        conn = db.get_connection()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        names = [r["name"] for r in tables]
        assert "tasks" in names
        assert "workers" in names

    def test_create_and_get_task(self):
        task_id = db.create_task("primes", {"limit": 100}, 4)
        assert task_id == 1

        task = db.get_task(task_id)
        assert task is not None
        assert task["algorithm"] == "primes"
        assert task["status"] == "pending"
        assert task["nodes"] == 4

    def test_get_next_pending_task(self):
        db.create_task("primes", {"limit": 100}, 2)
        db.create_task("stress", {"iterations": 1000}, 4)

        task = db.get_next_pending_task()
        assert task is not None
        assert task["algorithm"] == "primes"
        assert task["status"] == "running"

        task2 = db.get_next_pending_task()
        assert task2 is not None
        assert task2["algorithm"] == "stress"
        assert task2["status"] == "running"

        no_more = db.get_next_pending_task()
        assert no_more is None

    def test_complete_task(self):
        task_id = db.create_task("primes", {"limit": 100}, 2)
        db.complete_task(task_id, {"total_primes": 25}, 0.5)

        task = db.get_task(task_id)
        assert task["status"] == "completed"
        assert json.loads(task["result"])["total_primes"] == 25
        assert task["elapsed"] == 0.5

    def test_fail_task(self):
        task_id = db.create_task("primes", {"limit": 100}, 2)
        db.fail_task(task_id, "Something broke")

        task = db.get_task(task_id)
        assert task["status"] == "failed"
        assert "Something broke" in task["result"]

    def test_get_all_tasks_ordered(self):
        db.create_task("primes", {"limit": 100}, 2)
        db.create_task("stress", {"iterations": 1000}, 4)

        tasks = db.get_all_tasks()
        assert len(tasks) == 2
        assert tasks[0]["id"] == 2  # DESC order
        assert tasks[1]["id"] == 1

    def test_get_nonexistent_task(self):
        assert db.get_task(999) is None

    def test_register_and_get_workers(self):
        db.register_worker(0, "master")
        db.register_worker(1, "worker1")

        workers = db.get_workers()
        assert len(workers) == 2
        assert workers[0]["rank"] == 0
        assert workers[1]["host"] == "worker1"

    def test_register_worker_upsert(self):
        db.register_worker(0, "old_host")
        db.register_worker(0, "new_host")

        workers = db.get_workers()
        assert len(workers) == 1
        assert workers[0]["host"] == "new_host"
