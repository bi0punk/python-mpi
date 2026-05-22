import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

from master.db import init_db, create_task, get_next_pending_task, complete_task, fail_task, get_all_tasks, get_task, get_workers, register_worker
from common.models import TaskCreate, TaskResult, AlgorithmType

app = FastAPI(title="MPI Job Runner")

templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.on_event("startup")
def startup():
    init_db()


@app.get("/")
def dashboard(request: Request):
    tasks = get_all_tasks()
    for t in tasks:
        t["params"] = json.loads(t["params"])
        if t["result"]:
            t["result"] = json.loads(t["result"])
    workers = get_workers()
    return HTMLResponse(
        templates.TemplateResponse(
            "dashboard.html",
            {"request": request, "tasks": tasks, "workers": workers},
        )
    )


@app.post("/tasks")
def create_task_endpoint(task: TaskCreate):
    task_id = create_task(task.algorithm.value, task.params, task.nodes)
    return {"task_id": task_id, "status": "pending"}


@app.get("/tasks")
def list_tasks():
    tasks = get_all_tasks()
    for t in tasks:
        t["params"] = json.loads(t["params"])
        if t["result"]:
            t["result"] = json.loads(t["result"])
    return tasks


@app.get("/tasks/{task_id}")
def task_detail(task_id: int):
    task = get_task(task_id)
    if not task:
        return {"error": "Task not found"}, 404
    task["params"] = json.loads(task["params"])
    if task["result"]:
        task["result"] = json.loads(task["result"])
    return task


@app.get("/tasks/next/pending")
def next_pending():
    task = get_next_pending_task()
    if not task:
        return {"task": None}
    task["params"] = json.loads(task["params"])
    return {"task": task}


@app.post("/tasks/{task_id}/complete")
def complete_task_endpoint(task_id: int, result: TaskResult):
    complete_task(task_id, result.result, result.elapsed)
    return {"status": "completed"}


@app.post("/tasks/{task_id}/fail")
def fail_task_endpoint(task_id: int, error: str):
    fail_task(task_id, error)
    return {"status": "failed"}


@app.get("/workers")
def list_workers():
    return get_workers()


@app.post("/workers/ping")
def worker_ping(rank: int, host: str):
    register_worker(rank, host)
    return {"status": "ok"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
