def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def compute_chunk(start: int, end: int) -> list:
    return [n for n in range(start, end) if is_prime(n)]


def run(params: dict, rank: int, size: int) -> dict:
    limit = params.get("limit", 100000)
    chunk_size = limit // size
    chunk_start = rank * chunk_size
    chunk_end = limit if rank == size - 1 else (rank + 1) * chunk_size
    primes = compute_chunk(chunk_start, chunk_end)
    return {"prime_count": len(primes), "sample": primes[:10]}
