import sys
from unittest.mock import MagicMock


MPI_MOCK = MagicMock()
MPI_Mock = MagicMock()
MPI_Mock.COMM_WORLD = MagicMock()
MPI_Mock.COMM_WORLD.Get_rank = MagicMock(return_value=0)
MPI_Mock.COMM_WORLD.Get_size = MagicMock(return_value=4)
MPI_Mock.COMM_WORLD.send = MagicMock()
MPI_Mock.COMM_WORLD.recv = MagicMock()
MPI_Mock.COMM_WORLD.Is_null = MagicMock(return_value=False)

mpi_mock = MagicMock()
mpi_mock.MPI = MPI_Mock
mpi_mock.MPI.COMM_WORLD = MPI_Mock.COMM_WORLD

sys.modules["mpi4py"] = mpi_mock
sys.modules["mpi4py.MPI"] = MPI_Mock
