import numpy as np
import time

def row_major_traversal(matrix):
    rows, cols = matrix.shape
    s = 0.0
    for i in range(rows):        # outer over rows
        for j in range(cols):    # inner over contiguous dimension
            s += matrix[i, j]
    return s

def col_major_traversal(matrix):
    rows, cols = matrix.shape
    s = 0.0
    for j in range(cols):        # outer over columns
        for i in range(rows):    # inner over non-contiguous dimension
            s += matrix[i, j]
    return s

def benchmark(size=1500, repeats=3):
    print(f"Matrix size: {size} x {size}")
    a = np.random.rand(size, size).astype(np.float64)

    row_times = []
    col_times = []

    for r in range(repeats):
        # Row-major
        t0 = time.perf_counter()
        _ = row_major_traversal(a)
        t1 = time.perf_counter()
        row_times.append(t1 - t0)

        # Column-major
        t0 = time.perf_counter()
        _ = col_major_traversal(a)
        t1 = time.perf_counter()
        col_times.append(t1 - t0)

    print("Row-major times (s):    ", row_times)
    print("Column-major times (s):", col_times)
    print("Average row-major time:    {:.4f} s".format(sum(row_times) / len(row_times)))
    print("Average column-major time: {:.4f} s".format(sum(col_times) / len(col_times)))
    print("Speedup (col / row):       {:.2f}x slower".format(
        (sum(col_times) / len(col_times)) / (sum(row_times) / len(row_times))
    ))

if __name__ == "__main__":
    benchmark(size=900, repeats=3)
