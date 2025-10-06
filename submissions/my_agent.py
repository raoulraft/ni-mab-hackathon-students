# ================================================================
# TEMPLATE AGENT for the MAB Hackathon
# ------------------------------------------------
# This file defines a simple epsilon-greedy agent.
# Students will copy this structure and modify it to
# implement new exploration strategies or learning rules.
# ================================================================

import numpy as np, random

# --- Metadata for automatic leaderboard reporting ---
TEAM_NAME = "TEAM_NAME_HERE"     # ← students must change this
ALGO_NAME = "ChangeIt"        # ← and give their algorithm a clear name

# Factory function used by the harness to create the agent
def create_agent(n_arms: int, seed: int):
    """
    This function must return an initialized agent.
    It allows the evaluation harness to instantiate
    many independent copies of the agent.
    """
    return BanditAgent(n_arms, seed)


# ================================================================
# CLASS: BanditAgent
# ------------------------------------------------
# A simple epsilon-greedy bandit learner.
# ================================================================
class BanditAgent:
    def __init__(self, n_arms: int, seed: int, epsilon: float = 0.5):
        """
        n_arms:  number of available actions (arms)
        seed:    random seed for reproducibility
        epsilon: probability of taking a random (exploratory) action
        """
        self.n = int(n_arms)
        self.epsilon = float(epsilon)

        # Two RNGs: one for Python random, one for NumPy
        self.py = random.Random(seed)
        self.np = np.random.default_rng(seed)

        # Initialize learning state
        self.reset(seed)

    def reset(self, seed: int | None = None):
        """
        Called at the start of each new episode.
        Reinitializes RNGs and value estimates.
        """
        if seed is not None:
            self.py = random.Random(seed)
            self.np = np.random.default_rng(seed)

        # counts[i] = number of times arm i was pulled
        self.counts = np.zeros(self.n, dtype=int)

        # Q[i] = estimated mean reward of arm i
        self.Q = np.zeros(self.n, dtype=float)

    def select_arm(self, context=None) -> int:
        """
        Choose which arm to play next.
        - With probability epsilon: pick a random arm (exploration)
        - Otherwise: pick the arm with the highest Q estimate (exploitation)
        """
        # Random exploration
        if self.py.random() < self.epsilon:
            return self.py.randrange(self.n)

        # Greedy exploitation
        maxQ = np.max(self.Q)
        best = np.where(self.Q == maxQ)[0]
        # Break ties randomly among equally good arms
        return int(self.np.choice(best))

    def update(self, arm: int, reward: float, info: dict | None = None):
        """
        After playing an arm, update its estimated value.
        Incremental mean formula:
        Q_new = Q_old + (r - Q_old) / N
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n
