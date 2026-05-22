import time


def run(params: dict, rank: int, size: int) -> dict:
    iterations = params.get("iterations", 10_000_000)
    chunk = iterations // size
    start = time.perf_counter()
    x = 0
    for i in range(chunk):
        x += i * i
        x ^= (x >> 13) & 0xFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFF
        x ^= (x >> 5) & 0xFFFFFFFF
    elapsed = time.perf_counter() - start
    return {"ops": chunk, "elapsed": elapsed, "ops_per_sec": chunk / elapsed}
