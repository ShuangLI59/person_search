import itertools

from mpi4py import MPI


def mpi_dispatch(num_jobs, num_workers, worker_id):
    jobs_per_worker = num_jobs // num_workers
    start = worker_id * jobs_per_worker
    end = num_jobs if worker_id == num_workers-1 else start + jobs_per_worker
    return start, end


def mpi_collect(mpi_comm, mpi_rank, data):
    if isinstance(data, list):
        data = mpi_comm.gather(data, root=0)
        if mpi_rank == 0:
            data = list(itertools.chain.from_iterable(data))
    elif isinstance(data, dict):
        for k, v in data.iteritems():
            data[k] = mpi_collect(mpi_comm, mpi_rank, v)
    else:
        raise ValueError("Cannot collect result of " + type(data))
    return data