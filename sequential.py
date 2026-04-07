"""
sequential.py - Sequential Quicksort Implementation
"""
import time
import random


def quicksort(arr):
    """Standard recursive quicksort."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)


def generate_dataset(n: int, dataset_type: str = "random") -> list:
    """Generate dataset of given type and size."""
    if dataset_type == "random":
        return [random.randint(0, 100000) for _ in range(n)]
    elif dataset_type == "sorted":
        return list(range(n))
    elif dataset_type == "reverse":
        return list(range(n, 0, -1))
    elif dataset_type == "nearly_sorted":
        base = list(range(n))
        swaps = max(1, n // 20)
        for _ in range(swaps):
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            base[i], base[j] = base[j], base[i]
        return base
    else:
        return [random.randint(0, 100000) for _ in range(n)]


def run_sequential(data: list) -> tuple:
    """Run sequential quicksort and return (sorted_result, elapsed_time)."""
    start = time.perf_counter()
    result = quicksort(data)
    elapsed = time.perf_counter() - start
    return result, elapsed
