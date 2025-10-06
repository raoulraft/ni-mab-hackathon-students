import numpy as np

class SimpleBandit:
    """
    A very small, self-contained N-armed bandit environment.
    Each arm has an unknown reward distribution (mean = q*).
    When you pull an arm, you get a noisy reward around its true mean.
    """

    def __init__(self, n_arms=10, seed=0):
        """
        Initialize the bandit environment.
        - n_arms: number of actions (bandit arms).
        - seed: used for reproducibility.
        """
        self.np = np.random.default_rng(seed)           # independent random generator
        self.q_star = self.np.normal(0.0, 1.0, size=n_arms)  # true reward means for each arm
        self.n_arms = n_arms                            # number of possible arms
        self.t = 0                                      # time step counter

    def step(self, arm: int):
        """
        Execute one action (pull one arm).
        - arm: index of the chosen arm.
        Returns:
          r: reward sampled from Normal(q_star[arm], 1)
          info: dict with metadata (here, only the current time step)
        """
        self.t += 1                                     # increment time step
        r = float(self.np.normal(self.q_star[arm], 1.0)) # reward = true mean + Gaussian noise
        return r, {"t": self.t}
