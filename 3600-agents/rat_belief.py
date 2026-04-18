import numpy as np
import math

class RatBelief:

    def __init__(self, transition_matrix):
        self.T = transition_matrix
        self.reset_rat()

    def reset_rat(self):
        start_state = np.zeros(64)
        start_state[0] = 1.0
        # Precompute 1000-step transition for the rat's headstart
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.belief = start_state @ T_1000

    def predict(self):
        self.belief = self.belief @ self.T

    def update(self, noise, dist_estimate, board):
        new_belief = np.zeros(64)
        px, py = board.player_worker.get_location()

        for i in range(64):
            if self.belief[i] < 1e-5:
                continue

            x, y = i % 8, i // 8
            true_dist = abs(px - x) + abs(py - y)

            dist_prob = self.distance_likelihood(true_dist, dist_estimate)
            if dist_prob == 0:
                continue

            if (board._blocked_mask >> i) & 1:
                cell_type = "BLOCKED"
            elif (board._carpet_mask >> i) & 1:
                cell_type = "CARPET"
            elif (board._primed_mask >> i) & 1:
                cell_type = "PRIMED"
            else:
                cell_type = "SPACE"

            noise_prob = self.noise_likelihood(cell_type, noise)
            if noise_prob == 0:
                continue

            new_belief[i] = self.belief[i] * dist_prob * noise_prob

        total = np.sum(new_belief)
        if total > 0:
            self.belief = new_belief / total
        else:
            self.belief = np.ones(64) / 64

    def distance_likelihood(self, true_d, observed_d):
        # 🔥 FIXED: The game clamps negative distance rolls to 0
        if true_d == 0 and observed_d == 0:
            return 0.82  

        diff = observed_d - true_d
        if diff == -1: return 0.12
        if diff == 0:  return 0.70
        if diff == 1:  return 0.12
        if diff == 2:  return 0.06
        return 0.0

    def noise_likelihood(self, cell_type, noise):
        # 🔥 FIXED: Accurately mapped to the assignment's exact probabilities
        if cell_type == "BLOCKED":
            probs = {"squeak": 0.5, "scratch": 0.3,  "squeal": 0.2}
        elif cell_type == "SPACE":
            probs = {"squeak": 0.7, "scratch": 0.15, "squeal": 0.15}
        elif cell_type == "PRIMED":
            probs = {"squeak": 0.1, "scratch": 0.8,  "squeal": 0.1}
        elif cell_type == "CARPET":
            probs = {"squeak": 0.1, "scratch": 0.1,  "squeal": 0.8}
        
        return probs.get(noise, 0.0)

    def get_most_likely(self):
        idx = np.argmax(self.belief)
        return (idx % 8, idx // 8), self.belief[idx]

    def get_top_k(self, k=3):
        indices = np.argsort(self.belief)[-k:][::-1]
        return [((i % 8, i // 8), self.belief[i]) for i in indices]