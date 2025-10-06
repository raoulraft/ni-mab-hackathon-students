import numpy as np, random

TEAM_NAME = "TEAM_NAME_HERE"
ALGO_NAME = "MyEpsGreedy"

def create_agent(n_arms: int, seed: int):
    return BanditAgent(n_arms, seed)

class BanditAgent:
    def __init__(self, n_arms: int, seed: int, epsilon: float=0.1):
        self.n = int(n_arms); self.epsilon = float(epsilon)
        self.py = random.Random(seed); self.np = np.random.default_rng(seed)
        self.reset(seed)
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.py = random.Random(seed); self.np = np.random.default_rng(seed)
        self.counts = np.zeros(self.n, dtype=int)
        self.Q = np.zeros(self.n, dtype=float)
    def select_arm(self, context=None) -> int:
        if self.py.random() < self.epsilon:
            return self.py.randrange(self.n)
        maxQ = np.max(self.Q); best = np.where(self.Q == maxQ)[0]
        return int(self.np.choice(best))
    def update(self, arm: int, reward: float, info: dict | None = None):
        self.counts[arm] += 1; n = self.counts[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n
