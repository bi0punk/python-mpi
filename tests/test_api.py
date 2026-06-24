import json
import pytest
from fastapi.testclient import TestClient
from master.app import app
from common.models import AlgorithmType


@pytest.fixture
def client(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    import master.db as db_module
    monkeypatch.setattr(db_module, "DB_PATH", db_path)
    db_module.init_db()
    with TestClient(app) as c:
        yield c


class TestAPI:
    def test_dashboard_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_create_task(self, client):
        resp = client.post(
            "/tasks",
            json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 4},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == 1
        assert data["status"] == "pending"

    def test_list_tasks_empty(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_tasks_after_creation(self, client):
        client.post("/tasks", json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 2})
        client.post("/tasks", json={"algorithm": "stress", "params": {"iterations": 1000}, "nodes": 4})

        resp = client.get("/tasks")
        tasks = resp.json()
        assert len(tasks) == 2

    def test_task_detail(self, client):
        client.post("/tasks", json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 2})
        resp = client.get("/tasks/1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "primes"
        assert data["status"] == "pending"

    def test_task_detail_not_found(self, client):
        resp = client.get("/tasks/999")
        assert resp.status_code == 404
        assert resp.json() == {"error": "Task not found"}

    def test_next_pending_task(self, client):
        client.post("/tasks", json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 2})

        resp = client.get("/tasks/next/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task"]["id"] == 1
        assert data["task"]["status"] == "running"

    def test_next_pending_task_empty(self, client):
        resp = client.get("/tasks/next/pending")
        assert resp.status_code == 200
        assert resp.json() == {"task": None}

    def test_complete_task(self, client):
        client.post("/tasks", json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 2})

        resp = client.post(
            "/tasks/1/complete",
            json={"task_id": 1, "result": {"total_primes": 25}, "elapsed": 0.5, "worker_count": 2},
        )
        assert resp.status_code == 200
        assert resp.json() == {"status": "completed"}

        task = client.get("/tasks/1").json()
        assert task["status"] == "completed"

    def test_fail_task(self, client):
        client.post("/tasks", json={"algorithm": "primes", "params": {"limit": 100}, "nodes": 2})

        resp = client.post("/tasks/1/fail?error=test+error")
        assert resp.status_code == 200
        assert resp.json() == {"status": "failed"}

        task = client.get("/tasks/1").json()
        assert task["status"] == "failed"

    def test_worker_ping_and_list(self, client):
        resp = client.post("/workers/ping?rank=0&host=master")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

        resp = client.get("/workers")
        assert resp.status_code == 200
        workers = resp.json()
        assert len(workers) == 1
        assert workers[0]["rank"] == 0
        assert workers[0]["host"] == "master"

    def test_health_check_via_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_create_task_with_invalid_algorithm(self, client):
        resp = client.post(
            "/tasks",
            json={"algorithm": "invalid", "params": {}, "nodes": 1},
        )
        assert resp.status_code == 422
