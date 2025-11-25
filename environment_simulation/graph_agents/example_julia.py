import time
import numpy as np
from numba import njit
from numba.typed import List


# Numba-accelerated update
@njit
def evolve_numba(beliefs, neighbors, alpha):
    """
    beliefs: (N, 2) float64
    neighbors: List of 1D int64 arrays (typed list)
    """
    num_nodes = beliefs.shape[0]
    new_beliefs = beliefs.copy()

    for i in range(num_nodes):
        # toy "contagion": sum neighbor p(state=1)
        s = 0.0
        nbrs = neighbors[i]
        for k in range(nbrs.shape[0]):
            nb = nbrs[k]
            s += beliefs[nb, 1]

        # simple update rule just for testing
        p1 = beliefs[i, 1] + alpha * s
        if p1 < 0.0:
            p1 = 0.0
        if p1 > 1.0:
            p1 = 1.0

        new_beliefs[i, 1] = p1
        new_beliefs[i, 0] = 1.0 - p1

    return new_beliefs


def main():
    # toy graph: chain 0-1-2-...-N-1
    N = 10_000
    alpha = 0.001

    beliefs = np.zeros((N, 2), dtype=np.float64)
    beliefs[:, 1] = 0.2
    beliefs[:, 0] = 0.8

    # Python list-of-lists adjacency
    neighbors_py = [[] for _ in range(N)]
    for i in range(N):
        if i > 0:
            neighbors_py[i].append(i - 1)
        if i < N - 1:
            neighbors_py[i].append(i + 1)

    # *** CRITICAL: convert to numba.typed.List of np.int64 arrays ***
    neighbors_nb = List()
    for row in neighbors_py:
        neighbors_nb.append(np.array(row, dtype=np.int64))

    # Plain Python baseline
    def evolve_python(beliefs, neighbors, alpha):
        num_nodes = beliefs.shape[0]
        new_beliefs = beliefs.copy()
        for i in range(num_nodes):
            s = 0.0
            for nb in neighbors[i]:
                s += beliefs[nb, 1]
            p1 = beliefs[i, 1] + alpha * s
            p1 = min(max(p1, 0.0), 1.0)
            new_beliefs[i, 1] = p1
            new_beliefs[i, 0] = 1.0 - p1
        return new_beliefs

    # --- timing plain Python ---
    b_py = beliefs.copy()
    t0 = time.time()
    b_py = evolve_python(b_py, neighbors_py, alpha)
    t1 = time.time()
    print(f"Plain Python: {t1 - t0:.4f} sec")
    time_python = t1 - t0
    # --- first Numba call (includes compilation cost) ---
    b_nb = beliefs.copy()
    t0 = time.time()
    b_nb = evolve_numba(b_nb, neighbors_nb, alpha)
    t1 = time.time()
    print(f"Numba (first call, with compile): {t1 - t0:.4f} sec")

    # --- second Numba call (cached, real speed) ---
    t0 = time.time()
    b_nb = evolve_numba(b_nb, neighbors_nb, alpha)
    t1 = time.time()
    time_numba = t1 - t0
    print(f"Numba (second call, hot): {t1 - t0:.4f} sec")

    # sanity check
    print("Example node probs (Python vs Numba):")
    for i in range(3):
        print(i, b_py[i], b_nb[i])

    print(f"Python: {time_python:.4f} sec, Numba: {time_numba:.4f} sec")
    print(f"Speedup: {time_python / time_numba:.6f}x")
if __name__ == "__main__":
    main()
