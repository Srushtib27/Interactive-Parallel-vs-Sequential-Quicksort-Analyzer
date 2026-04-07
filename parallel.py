"""
parallel.py - Parallel Quicksort using MPI (mpi4py)
Run with: mpirun -np <N> python parallel.py --size <SIZE> --dtype <TYPE>
"""
import sys
import time
import random
import argparse

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def generate_data(n, dtype):
    if dtype == "sorted":
        return list(range(n))
    elif dtype == "reverse":
        return list(range(n, 0, -1))
    elif dtype == "nearly_sorted":
        base = list(range(n))
        for _ in range(max(1, n // 20)):
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            base[i], base[j] = base[j], base[i]
        return base
    else:
        return [random.randint(0, 100000) for _ in range(n)]


def run_mpi_sort(n, dtype):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Root generates data and splits it
    if rank == 0:
        data = generate_data(n, dtype)
        # Split into equal chunks
        chunk_size = n // size
        chunks = []
        for i in range(size):
            start = i * chunk_size
            end = start + chunk_size if i < size - 1 else n
            chunks.append(data[start:end])
    else:
        chunks = None

    # Barrier to sync before timing
    comm.Barrier()
    t_start = MPI.Wtime()

    # Scatter chunks to all processes
    local_chunk = comm.scatter(chunks, root=0)

    # Each process sorts its chunk
    local_sorted = quicksort(local_chunk)

    # Gather all sorted chunks at root
    all_sorted = comm.gather(local_sorted, root=0)

    if rank == 0:
        # Merge all sorted chunks
        final = []
        import heapq
        iterators = [iter(chunk) for chunk in all_sorted]
        heap = []
        for i, it in enumerate(iterators):
            try:
                val = next(it)
                heapq.heappush(heap, (val, i))
            except StopIteration:
                pass

        while heap:
            val, i = heapq.heappop(heap)
            final.append(val)
            try:
                next_val = next(iterators[i])
                heapq.heappush(heap, (next_val, i))
            except StopIteration:
                pass

        t_end = MPI.Wtime()
        elapsed = t_end - t_start
        print(f"PARALLEL_TIME:{elapsed:.6f}")
        print(f"PARALLEL_SIZE:{len(final)}")
        sys.stdout.flush()
    else:
        pass  # Worker processes exit cleanly


if __name__ == "__main__":
    if not MPI_AVAILABLE:
        print("ERROR:mpi4py not installed")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--dtype", type=str, default="random")
    args = parser.parse_args()

    run_mpi_sort(args.size, args.dtype)
