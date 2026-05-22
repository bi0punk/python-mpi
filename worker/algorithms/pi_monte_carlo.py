import random


def run(params: dict, rank: int, size: int) -> dict:
    points = params.get("points", 10_000_000)
    chunk = points // size
    inside = 0
    for _ in range(chunk):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    pi_approx = 4.0 * inside / chunk
    return {"inside": inside, "total": chunk, "pi_approx": pi_approx}
