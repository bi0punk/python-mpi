import sys
import time
import json
import socket
import urllib.request
import urllib.error

from mpi4py import MPI

from common.serializer import serialize, deserialize
from worker.algorithms import primes, pi_monte_carlo, stress

MASTER_URL = "http://localhost:8000"
TAG_WORK = 100
TAG_RESULT = 200

ALGORITHMS = {
    "primes": primes,
    "pi_monte_carlo": pi_monte_carlo,
    "stress": stress,
}


def poll_next_task():
    url = f"{MASTER_URL}/tasks/next/pending"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        data = json.loads(resp.read())
        return data.get("task")
    except Exception:
        return None


def report_result(task_id: int, result: dict, elapsed: float, worker_count: int):
    payload = json.dumps({
        "task_id": task_id,
        "result": result,
        "elapsed": elapsed,
        "worker_count": worker_count,
    }).encode()
    req = urllib.request.Request(
        f"{MASTER_URL}/tasks/{task_id}/complete",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[coordinator] Error reporting result: {e}")


def report_failure(task_id: int, error: str):
    payload = json.dumps({"error": error}).encode()
    req = urllib.request.Request(
        f"{MASTER_URL}/tasks/{task_id}/fail?error={error}",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        hostname = socket.gethostname()
        print(f"[coordinator] Rank 0 ({hostname}) — {size} workers total")

        while True:
            task = poll_next_task()
            if task is None:
                time.sleep(2)
                continue

            task_id = task["id"]
            algorithm = task["algorithm"]
            params = task["params"]
            print(f"[coordinator] Got task #{task_id}: {algorithm} params={params}")

            try:
                algo_module = ALGORITHMS.get(algorithm)
                if not algo_module:
                    report_failure(task_id, f"Unknown algorithm: {algorithm}")
                    continue

                start = time.perf_counter()

                for dest in range(1, size):
                    comm.send((task_id, algorithm, params, start), dest=dest, tag=TAG_WORK)

                partial = algo_module.run(params, rank, size)
                partials = [partial]
                for src in range(1, size):
                    partials.append(comm.recv(source=src, tag=TAG_RESULT))

                end = time.perf_counter()
                elapsed = end - start

                aggregated = {"workers": partials}
                if algorithm == "primes":
                    total_primes = sum(p["prime_count"] for p in partials)
                    aggregated["total_primes"] = total_primes
                elif algorithm == "pi_monte_carlo":
                    total_inside = sum(p["inside"] for p in partials)
                    total_points = sum(p["total"] for p in partials)
                    aggregated["pi_approx"] = 4.0 * total_inside / total_points
                    aggregated["total_points"] = total_points
                elif algorithm == "stress":
                    avg_ops = sum(p["ops_per_sec"] for p in partials) / len(partials)
                    aggregated["avg_ops_per_sec"] = avg_ops

                # TODO: replace with a proper sequential baseline measurement
                aggregated["speedup"] = 1.0

                report_result(task_id, aggregated, elapsed, size)
                print(f"[coordinator] Task #{task_id} done in {elapsed:.4f}s")

            except Exception as e:
                print(f"[coordinator] Task #{task_id} failed: {e}")
                report_failure(task_id, str(e))
    else:
        hostname = socket.gethostname()
        print(f"[worker] Rank {rank} ({hostname}) ready")
        while True:
            try:
                status = MPI.Status()
                data = comm.recv(source=0, tag=TAG_WORK, status=status)
                if status.Get_tag() != TAG_WORK:
                    continue
                task_id, algorithm, params, start_time = data
                algo_module = ALGORITHMS.get(algorithm)
                if algo_module:
                    result = algo_module.run(params, rank, size)
                    comm.send(result, dest=0, tag=TAG_RESULT)
            except Exception:
                pass


if __name__ == "__main__":
    main()
