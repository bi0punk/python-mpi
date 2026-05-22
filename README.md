# MPI Job Runner

**Sistema de cola de tareas distribuidas con MPI y dashboard web.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![MPI](https://img.shields.io/badge/MPI-mpi4py-00618A)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ¿Qué es y para qué sirve?

**MPI Job Runner** es un laboratorio educativo y práctico de **cómputo distribuido**. Permite:

- Encolar tareas de cómputo intensivo a través de una **API REST**
- Ejecutarlas en **paralelo** usando MPI (Message Passing Interface) en múltiples nodos/procesos
- Monitorear el progreso y resultados desde un **dashboard web** en tiempo real
- Comparar rendimiento (speedup) entre ejecución single vs distribuida

Es ideal para:
- Aprender y experimentar con **paralelización de algoritmos**
- Probar **clusters MPI** locales o remotos
- Hacer **benchmarks** de rendimiento en CPUs multi-núcleo
- Servir como base para un **sistema de cómputo distribuido** real

---

## Arquitectura

```
                    ┌──────────────────────────────────┐
                    │        Master (FastAPI)           │
                    │  Puerto 8000 · SQLite · Dashboard │
                    └────────────┬─────────────────────┘
                                 │ HTTP (REST API)
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
     ┌────▼────┐           ┌────▼────┐           ┌────▼────┐
     │Worker   │           │Worker   │           │Worker   │
     │rank 0   │◄─────────►│rank 1   │◄─────────►│rank 2   │
     │coordina.│    MPI    │compute  │    MPI    │compute  │
     └─────────┘           └─────────┘           └─────────┘
```

### Componentes

| Componente | Rol | Tecnología |
|------------|-----|------------|
| **Master** | Servidor web que recibe tareas, las encola y expone resultados | FastAPI + SQLite |
| **Coordinator** (rank 0) | Worker líder: consulta tareas pendientes, coordina workers vía MPI, reporta resultados | Python + mpi4py |
| **Compute** (rank 1..N) | Workers esclavos: reciben trabajo del coordinator, ejecutan su chunk, devuelven resultado | Python + mpi4py |
| **Dashboard** | UI web con estadísticas, gráficos de speedup y tabla de tareas | Jinja2 + Chart.js |

### Flujo de una tarea

```
                  Master                   Coordinator (rank 0)      Compute (rank 1..N)
                    │                              │                        │
  POST /tasks       │                              │                        │
  ─────────────────►│                              │                        │
                    │   GET /tasks/next/pending     │                        │
                    │◄─────────────────────────────│                        │
                    │   {task}                      │                        │
                    │─────────────────────────────►│                        │
                    │                              │   MPI send params      │
                    │                              │───────────────────────►│
                    │                              │                        ├── compute chunk
                    │                              │   MPI gather results   │
                    │                              │◄───────────────────────│
                    │   POST /tasks/{id}/complete   │                        │
                    │◄─────────────────────────────│                        │
  Dashboard actualiza │                              │                        │
```

---

## Algoritmos incluidos

| Algoritmo | Descripción | Cómo se paraleliza |
|-----------|-------------|-------------------|
| `primes` | Cuenta números primos hasta N | Cada worker revisa `N/size` números |
| `pi_monte_carlo` | Aproximación de Pi lanzando puntos aleatorios | Cada worker lanza `N/size` puntos; se suman los `inside` |
| `stress` | Benchmark sintético de CPU (operaciones xor/shift) | Cada worker hace `N/size` iteraciones |

---

## Instalación

```bash
# 1. Clonar
git clone https://github.com/bi0punk/python-mpi
cd python-mpi

# 2. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar MPI
mpirun --version
python -c "from mpi4py import MPI; print(f'{MPI.COMM_WORLD.Get_size()} processes available')"
```

---

## Uso

### 1. Iniciar el sistema

```bash
# Opción A: Script automático
chmod +x run.sh
./run.sh

# Opción B: Manual (dos terminales)
# Terminal 1 - Master
python -m master.app

# Terminal 2 - Workers (4 procesos)
mpirun -np 4 --hostfile machines python -m worker.coordinator
```

### 2. Crear tareas vía API

```bash
# Primos hasta 100,000
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "primes", "params": {"limit": 100000}, "nodes": 4}'

# Pi Monte Carlo con 10 millones de puntos
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "pi_monte_carlo", "params": {"points": 10000000}, "nodes": 4}'

# Stress test
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "stress", "params": {"iterations": 50000000}, "nodes": 4}'
```

### 3. Ver el dashboard

Abrir en el navegador: [http://localhost:8000](http://localhost:8000)

### 4. API endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Dashboard web |
| `POST` | `/tasks` | Crear tarea |
| `GET` | `/tasks` | Listar todas las tareas |
| `GET` | `/tasks/{id}` | Detalle de una tarea |
| `GET` | `/workers` | Workers conectados |

---

## Ejemplo de respuesta

```json
// POST /tasks
{ "task_id": 1, "status": "pending" }

// GET /tasks/1 (después de completar)
{
  "id": 1,
  "algorithm": "pi_monte_carlo",
  "params": {"points": 10000000},
  "nodes": 4,
  "status": "completed",
  "result": {
    "pi_approx": 3.141592,
    "total_points": 10000000,
    "workers": [...],
    "speedup": 3.82
  },
  "elapsed": 0.4231,
  "created_at": "2025-05-22T12:00:00+00:00",
  "completed_at": "2025-05-22T12:00:01+00:00"
}
```

---

## Estructura del proyecto

```
python-mpi/
├── README.md
├── pyproject.toml
├── requirements.txt
├── machines                  # Hostfile MPI
├── run.sh                    # Script para levantar todo
├── .gitignore
│
├── common/                   # Código compartido
│   ├── __init__.py
│   ├── models.py             # Pydantic models
│   └── serializer.py         # Serialización pickle
│
├── master/                   # Servidor central
│   ├── __init__.py
│   ├── app.py                # FastAPI + rutas
│   ├── db.py                 # SQLite (tareas, workers)
│   └── templates/
│       └── dashboard.html    # Dashboard web con Chart.js
│
├── worker/                   # Código distribuido
│   ├── __init__.py
│   ├── coordinator.py        # Rank 0: coordina workers
│   └── algorithms/
│       ├── __init__.py
│       ├── primes.py         # Conteo de primos
│       ├── pi_monte_carlo.py # Aproximación de Pi
│       └── stress.py         # Benchmark sintético
│
└── tests/                    # (futuro)
    └── __init__.py
```

---

## Cómo agregar un nuevo algoritmo

1. Crear `worker/algorithms/mi_algoritmo.py`
2. Implementar función `run(params, rank, size) -> dict`
3. Registrar en `worker/coordinator.py` → diccionario `ALGORITHMS`

Ejemplo:

```python
# worker/algorithms/sum_squares.py
def run(params, rank, size):
    n = params.get("n", 1000)
    chunk = n // size
    start = rank * chunk
    end = n if rank == size - 1 else (rank + 1) * chunk
    total = sum(i * i for i in range(start, end))
    return {"partial_sum": total, "start": start, "end": end}
```

---

## Requisitos

- Python 3.10+
- OpenMPI o MPICH instalado en el sistema
- mpi4py (`pip install mpi4py`)

### Verificar MPI

```bash
# Linux
sudo apt install openmpi-bin openmpi-common libopenmpi-dev

# macOS
brew install open-mpi

# Verificar
mpirun --version
```

---

## Licencia

MIT
