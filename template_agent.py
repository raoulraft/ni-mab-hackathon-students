import numpy as np, random

# Metadata used by the evaluation harness
TEAM_NAME = "TEAM_NAME_HERE"          # <-- students should replace this with their name or team name
ALGO_NAME = "MyEpsGreedy (Change the name)"             # <-- algorithm nickname (appears on leaderboard)

def create_agent(n_arms: int, seed: int):
    """
    Factory function called by the simulator to create an agent instance.
    It passes the number of arms and a random seed.
    """
    return BanditAgent(n_arms, seed)


class BanditAgent:
    """
    A simple implementation of the ε-greedy algorithm for multi-armed bandits.

    Concept:
      - Each arm has an estimated value Q(a)
      - With probability ε, we explore (choose a random arm)
      - With probability (1−ε), we exploit (choose arm with highest Q)
      - After each reward, we update Q(a) as the running average.
    """

    def __init__(self, n_arms: int, seed: int, epsilon: float=0.1):
        """
        Initialize the agent.
        - n_arms: number of available actions.
        - seed: reproducibility seed.
        - epsilon: exploration probability (default = 0.1).
        """
        self.n = int(n_arms)
        self.epsilon = float(epsilon)
        self.py = random.Random(seed)                   # Python random for ε decisions
        self.np = np.random.default_rng(seed)           # NumPy RNG for tie-breaking
        self.reset(seed)                                # initialize internal state

    def reset(self, seed: int | None = None):
        """
        Reset internal statistics (e.g., between episodes).
        If a seed is given, the RNGs are also re-seeded.
        """
        if seed is not None:
            self.py = random.Random(seed)
            self.np = np.random.default_rng(seed)
        self.counts = np.zeros(self.n, dtype=int)       # number of times each arm was pulled
        self.Q = np.zeros(self.n, dtype=float)          # estimated value (mean reward) for each arm

    def select_arm(self, context=None) -> int:
        """
        Choose an arm to play according to ε-greedy rule:
        - With probability ε: random arm (exploration)
        - Otherwise: arm with highest estimated value (exploitation)
        """
        if self.py.random() < self.epsilon:             # exploration
            return self.py.randrange(self.n)
        maxQ = np.max(self.Q)                           # find best estimated value
        best = np.where(self.Q == maxQ)[0]              # all arms sharing the max value
        return int(self.np.choice(best))                # random tie-break among best

    def update(self, arm: int, reward: float, info: dict | None = None):
        """
        Update internal Q-value estimate after observing the reward.
        Uses the incremental mean formula:
          Q_new = Q_old + (reward - Q_old) / N(arm)
        """
        self.counts[arm] += 1                           # increment pull count for this arm
        n = self.counts[arm]
        self.Q[arm] += (reward - self.Q[arm]) / n       # incremental update rule
