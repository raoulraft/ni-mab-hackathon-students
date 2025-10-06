import numpy as np

class SimpleBandit:
    def __init__(self, n_arms=10, seed=0):
        self.np = np.random.default_rng(seed)
        self.q_star = self.np.normal(0.0, 1.0, size=n_arms)
        self.n_arms = n_arms
        self.t = 0
    def step(self, arm: int):
        self.t += 1
        r = float(self.np.normal(self.q_star[arm], 1.0))
        return r, {"t": self.t}
