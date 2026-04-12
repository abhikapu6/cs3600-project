"""Precompute transition-matrix-derived quantities at init time."""

import numpy as np

from game.enums import BOARD_SIZE, Cell, Noise
from game.rat import NOISE_PROBS, DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS, HEADSTART_MOVES

N = BOARD_SIZE * BOARD_SIZE  # 64


def jax_to_numpy(T_jax) -> np.ndarray:
    """Convert JAX transition matrix to numpy float32."""
    return np.array(T_jax, dtype=np.float32)


def compute_spawn_dist(T_np: np.ndarray) -> np.ndarray:
    """Compute stationary-ish distribution after 1000 moves from (0,0) via repeated squaring.

    spawn_dist = e_{0} @ T^1000.  We use repeated squaring on T:
    1000 = 0b1111101000, so ~10 squarings + multiplications.
    """
    # T^1000 via repeated squaring
    result = np.eye(N, dtype=np.float32)
    base = T_np.copy()
    exp = HEADSTART_MOVES
    while exp > 0:
        if exp & 1:
            result = result @ base
        base = base @ base
        exp >>= 1
    # spawn_dist = e_0 @ T^1000 = result[0, :]
    return result[0].copy()


def compute_manhattan_lut() -> np.ndarray:
    """Precompute manhattan distance between all pairs of cells.

    Returns: int array [64, 64] where lut[i, j] = manhattan(pos_i, pos_j).
    """
    lut = np.zeros((N, N), dtype=np.int32)
    for i in range(N):
        y_i, x_i = divmod(i, BOARD_SIZE)
        # Note: index = y * BOARD_SIZE + x, but pos = (x, y).
        # rat.py: _pos_to_index: pos[1] * BOARD_SIZE + pos[0] => index = y*8+x
        # So for index i: y_i = i // 8, x_i = i % 8
        for j in range(N):
            y_j, x_j = divmod(j, BOARD_SIZE)
            lut[i, j] = abs(x_i - x_j) + abs(y_i - y_j)
    return lut


def compute_dist_likelihood() -> np.ndarray:
    """Precompute P(observed_distance | true_manhattan_distance).

    observed can be 0..14 (max manhattan on 8x8 is 14), true can be 0..14.
    The observed distance = true + offset, clamped to max(0, ...).

    Returns: float32 array [15, 15] where dist_lik[observed, true_dist] = P(obs | true).
    """
    max_dist = 2 * (BOARD_SIZE - 1)  # 14
    lik = np.zeros((max_dist + 3, max_dist + 1), dtype=np.float32)  # +3 for offset +2 safety

    for true_d in range(max_dist + 1):
        for offset, prob in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS):
            obs = true_d + offset
            if obs < 0:
                obs = 0  # clamping as in rat.py:125
            if obs < lik.shape[0]:
                lik[obs, true_d] += prob

    return lik


def compute_noise_likelihood() -> np.ndarray:
    """Precompute P(noise | cell_type) for each cell type.

    Returns: float32 array [4, 3] where noise_lik[cell_type, noise] = P(noise | cell_type).
    Cell types: SPACE=0, PRIMED=1, CARPET=2, BLOCKED=3.
    Noise: SQUEAK=0, SCRATCH=1, SQUEAL=2.
    """
    lik = np.zeros((4, 3), dtype=np.float32)
    cell_type_map = {
        Cell.SPACE: 0,
        Cell.PRIMED: 1,
        Cell.CARPET: 2,
        Cell.BLOCKED: 3,
    }
    for cell_type, probs in NOISE_PROBS.items():
        idx = cell_type_map[cell_type]
        for noise_idx in range(3):
            lik[idx, noise_idx] = probs[noise_idx]
    return lik


class Precomputed:
    """Holds all precomputed data derived from the transition matrix."""

    def __init__(self, T_jax):
        self.T_np = jax_to_numpy(T_jax)
        self.spawn_dist = compute_spawn_dist(self.T_np)
        self.manhattan_lut = compute_manhattan_lut()
        self.dist_lik = compute_dist_likelihood()
        self.noise_lik = compute_noise_likelihood()
