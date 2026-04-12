"""HMM forward-filter for rat position belief state."""

import os
import numpy as np

from game.enums import BOARD_SIZE, Cell, Noise

N = BOARD_SIZE * BOARD_SIZE  # 64

_DEBUG = os.environ.get("ALBRECHT_DEBUG") == "1"

# Map Cell enum to noise_lik row index
_CELL_TO_IDX = {
    Cell.SPACE: 0,
    Cell.PRIMED: 1,
    Cell.CARPET: 2,
    Cell.BLOCKED: 3,
}


def _pos_to_index(pos):
    """(x, y) -> flat index (y * 8 + x), matching rat.py convention."""
    return pos[1] * BOARD_SIZE + pos[0]


def _index_to_pos(idx):
    """Flat index -> (x, y)."""
    return (idx % BOARD_SIZE, idx // BOARD_SIZE)


class RatBelief:
    """Maintains a probability distribution over the 64 board cells for the rat's location."""

    def __init__(self, precomputed):
        self.pre = precomputed
        # Start with uniform prior (we don't know where the rat is initially)
        self.b = np.full(N, 1.0 / N, dtype=np.float32)

    def predict(self):
        """Advance belief by one rat transition step: b = b @ T."""
        self.b = self.b @ self.pre.T_np

    def update_sensor(self, noise, dist, worker_pos, board):
        """Bayesian update from sensor reading (noise type, observed manhattan distance).

        Args:
            noise: Noise enum value (SQUEAK=0, SCRATCH=1, SQUEAL=2)
            dist: observed (noisy) manhattan distance (int >= 0)
            worker_pos: (x, y) of our worker BEFORE moving this turn
            board: Board object (to read cell types for noise likelihood)
        """
        noise_idx = int(noise)
        worker_idx = _pos_to_index(worker_pos)

        # --- Noise likelihood: P(noise | cell_type_at_rat_pos) for each cell ---
        noise_vec = np.empty(N, dtype=np.float32)
        for i in range(N):
            pos = _index_to_pos(i)
            cell_type = board.get_cell(pos)
            cell_idx = _CELL_TO_IDX.get(cell_type, 0)  # default SPACE
            noise_vec[i] = self.pre.noise_lik[cell_idx, noise_idx]

        # --- Distance likelihood: P(observed_dist | true_manhattan(worker, rat_cell)) ---
        # manhattan_lut[worker_idx, :] gives true distance from worker to each cell
        true_dists = self.pre.manhattan_lut[worker_idx]
        dist_vec = np.empty(N, dtype=np.float32)
        dist_lik = self.pre.dist_lik
        obs = min(dist, dist_lik.shape[0] - 1)
        for i in range(N):
            td = true_dists[i]
            if td < dist_lik.shape[1]:
                dist_vec[i] = dist_lik[obs, td]
            else:
                dist_vec[i] = 0.0

        # Combined update
        self.b *= noise_vec * dist_vec
        total = self.b.sum()
        if total > 0:
            self.b /= total
        else:
            # Degenerate case — reset to uniform
            self.b[:] = 1.0 / N

        if _DEBUG:
            assert abs(self.b.sum() - 1.0) < 1e-4, f"Belief not normalized: sum={self.b.sum()}"

    def update_search(self, loc, hit):
        """Update belief based on a search action result.

        Args:
            loc: (x, y) search location, or None if no search was made
            hit: True if rat was found, False if miss, or False/None if no search
        """
        if loc is None:
            return

        idx = _pos_to_index(loc)

        if hit:
            # Rat was caught and respawned — reset belief to spawn distribution
            self.b = self.pre.spawn_dist.copy()
        else:
            # Rat was NOT at loc — zero out that cell and renormalize
            self.b[idx] = 0.0
            total = self.b.sum()
            if total > 0:
                self.b /= total
            else:
                self.b[:] = 1.0 / N

    def clone(self):
        """Return a lightweight copy for branching inside search tree."""
        new = RatBelief.__new__(RatBelief)
        new.pre = self.pre
        new.b = self.b.copy()
        return new

    def argmax(self):
        """Return (x, y) of the most likely rat position."""
        idx = int(np.argmax(self.b))
        return _index_to_pos(idx)

    def max_prob(self):
        """Return the probability of the most likely cell."""
        return float(np.max(self.b))

    def ev_best_search(self):
        """Expected value of the best possible SEARCH action.

        EV = max_c [ RAT_BONUS * b[c] - RAT_PENALTY * (1 - b[c]) ]
           = max_c [ (RAT_BONUS + RAT_PENALTY) * b[c] - RAT_PENALTY ]
           = 6 * max(b) - 2
        """
        return 6.0 * float(np.max(self.b)) - 2.0
