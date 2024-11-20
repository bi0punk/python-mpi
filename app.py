from mpi4py import MPI
import time

def sum_of_squares(start, end):
    """Calcula la suma de los cuadrados de los números en un rango."""
    return sum(i ** 2 for i in range(start, end))

def single_machine_calculation(total_numbers):
    """Cálculo secuencial en una sola máquina y mide el tiempo."""
    start_time = time.time()
    result = sum_of_squares(0, total_numbers)
    end_time = time.time()
    print(f"[Single Machine] Resultado: {result}, Tiempo: {end_time - start_time:.4f} segundos")

def distributed_calculation(total_numbers):
    """Cálculo distribuido utilizando MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dividir el rango de números para cada proceso
    chunk_size = total_numbers // size
    start = rank * chunk_size
    end = total_numbers if rank == size - 1 else (rank + 1) * chunk_size

    # Nodo maestro mide el tiempo total
    if rank == 0:
        start_time = MPI.Wtime()

    # Cada proceso realiza su cálculo
    partial_result = sum_of_squares(start, end)

    # Reducir resultados al nodo maestro
    result = comm.reduce(partial_result, op=MPI.SUM, root=0)

    # Nodo maestro muestra el resultado y el tiempo total
    if rank == 0:
        end_time = MPI.Wtime()
        print(f"[Distributed] Resultado: {result}, Tiempo: {end_time - start_time:.4f} segundos")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python app.py modo total_numbers")
        print("modos: single o distributed")
        sys.exit(1)

    mode = sys.argv[1]
    total_numbers = int(sys.argv[2])

    if mode == "single":
        single_machine_calculation(total_numbers)
    elif mode == "distributed":
        distributed_calculation(total_numbers)
    else:
        print("Modo desconocido. Usa 'single' o 'distributed'.")
