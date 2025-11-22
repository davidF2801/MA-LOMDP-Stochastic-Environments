import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

def enum_binary_states(K):
    """Enumerate all 2^K binary states as (2^K, K) array of 0/1."""
    S = 1 << K
    states = np.zeros((S, K), dtype=np.uint8)
    for idx in range(S):
        for bit in range(K):
            states[idx, bit] = (idx >> bit) & 1
    return states

def random_markov_matrix(S, self_weight=0.9, seed=None):
    """Generate a random row-stochastic matrix with strong diagonal (slow transitions)."""
    rng = np.random.default_rng(seed)
    P = rng.random((S, S)) * (1 - self_weight)
    np.fill_diagonal(P, self_weight)
    P /= P.sum(axis=1, keepdims=True)
    return P

def simulate(rows=3, cols=3, steps=100, seed=0):
    K = rows * cols
    S = 1 << K
    print(f"Grid {rows}x{cols} → {S} joint states")

    rng = np.random.default_rng(seed)
    states = enum_binary_states(K)
    P = random_markov_matrix(S, self_weight=0.9, seed=seed)

    # Start from a random joint state
    curr_idx = int(rng.integers(0, S))
    curr_state = states[curr_idx].reshape(rows, cols)

    # Store all sampled realizations for visualization
    samples = [curr_state.copy()]
    for _ in range(steps):
        curr_idx = rng.choice(S, p=P[curr_idx])  # sample next state
        curr_state = states[curr_idx].reshape(rows, cols)
        samples.append(curr_state.copy())

    # ---------------- ANIMATION ----------------
    fig, ax = plt.subplots(figsize=(4, 4))
    img = ax.imshow(samples[0], vmin=0, vmax=1, cmap="binary", interpolation="nearest")
    ax.set_title("t = 0")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    def update(frame):
        img.set_data(samples[frame])
        ax.set_title(f"t = {frame}")
        return (img,)

    ani = animation.FuncAnimation(fig, update, frames=len(samples),
                                  interval=200, blit=True, repeat=False)
    plt.show()

if __name__ == "__main__":
    simulate(rows=3, cols=3, steps=60, seed=1)
