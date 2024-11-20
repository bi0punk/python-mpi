from mpi4py import MPI
import time

def is_prime(n):
    """Verifica si un número es primo."""
    if n <= 1:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def generate_first_n_primes(n):
    """Genera los primeros n números primos."""
    primes = []
    num = 2
    while len(primes) < n:
        if is_prime(num):
            primes.append(num)
        num += 1
    return primes

def single_machine_calculation(total_primes):
    """Cálculo secuencial en una sola máquina y mide el tiempo."""
    start_time = time.time()
    primes = generate_first_n_primes(total_primes)
    end_time = time.time()
    print(f"[Single Machine] Primeros {total_primes} primos: {primes}")
    print(f"Tiempo: {end_time - start_time:.4f} segundos")

def distributed_calculation(total_primes):
    """Cálculo distribuido utilizando MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dividir la tarea entre los procesos
    chunk_size = total_primes // size
    start_index = rank * chunk_size
    end_index = (rank + 1) * chunk_size if rank != size - 1 else total_primes

    # Nodo maestro mide el tiempo total
    if rank == 0:
        start_time = MPI.Wtime()

    # Generación de primos en un rango específico
    primes = []
    num = 2
    prime_count = 0
    while prime_count < total_primes:
        if is_prime(num):
            if start_index <= prime_count < end_index:
                primes.append(num)
            prime_count += 1
        num += 1

    # Recoger los resultados en el nodo maestro
    all_primes = comm.gather(primes, root=0)

    # Nodo maestro muestra el resultado y el tiempo total
    if rank == 0:
        # Aplanar la lista de resultados
        all_primes = [prime for sublist in all_primes for prime in sublist]
        all_primes.sort()  # Asegurar que los primos estén ordenados
        end_time = MPI.Wtime()
        print(f"[Distributed] Primeros {total_primes} primos: {all_primes}")
        print(f"Tiempo: {end_time - start_time:.4f} segundos")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python primos.py modo total_primes")
        print("modos: single o distributed")
        sys.exit(1)

    mode = sys.argv[1]
    total_primes = int(sys.argv[2])

    if mode == "single":
        single_machine_calculation(total_primes)
    elif mode == "distributed":
        distributed_calculation(total_primes)
    else:
        print("Modo desconocido. Usa 'single' o 'distributed'.")
