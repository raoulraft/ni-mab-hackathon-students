import importlib.util, os, sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from simple_env import SimpleBandit

# ---------- config you can tweak ----------
EPISODES = 100          # more episodes -> less noise
STEPS = 2000            # longer horizon shows learning
SMOOTH_WINDOW = 50      # moving average window for nicer curves
N_ARMS = 10
SEED = 7
# ------------------------------------------

def load_agent(module_path):
    spec = importlib.util.spec_from_file_location("agent_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod

def moving_average(x, w):
    if w <= 1: return x
    c = np.convolve(x, np.ones(w)/w, mode="valid")
    # pad to original length for nice plotting
    pad_left = w-1
    return np.concatenate([np.full(pad_left, c[0]), c])

def run_episode(agent_factory, steps, seed):
    env = SimpleBandit(n_arms=N_ARMS, seed=seed)
    agent = agent_factory(env.n_arms, seed)
    agent.reset(seed)

    rewards = np.zeros(steps)
    # regret needs the best-possible expected reward this episode
    best_q = np.max(env.q_star)  # optimal mean reward this episode
    regret = np.zeros(steps)

    cum_regret = 0.0
    for t in range(steps):
        a = agent.select_arm()
        r, info = env.step(a)
        agent.update(a, r, info)
        rewards[t] = r
        cum_regret += (best_q - r)   # instantaneous regret approx
        regret[t] = cum_regret
    return rewards, regret

def evaluate_agent(agent_factory, episodes, steps, base_seed):
    # episode-wise average reward curve & regret
    avg_reward = np.zeros(steps)
    avg_regret = np.zeros(steps)
    for ep in range(episodes):
        rewards, regret = run_episode(agent_factory, steps, base_seed + ep)
        avg_reward += rewards
        avg_regret += regret
    avg_reward /= episodes
    avg_regret /= episodes
    return avg_reward, avg_regret

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..", "submissions")
    agent_files = sorted([f for f in os.listdir(base_dir) if f.endswith(".py") and not f.startswith("__")])
    if not agent_files:
        print("❌ No agents found in 'submissions/'! Create my_agent.py first.")
        sys.exit(1)
    print(f"✅ Found {len(agent_files)} agent(s): {', '.join(agent_files)}")

    results = {}
    for file in agent_files:
        path = os.path.join(base_dir, file)
        mod = load_agent(path)
        t0 = time.time()
        avg_reward, avg_regret = evaluate_agent(mod.create_agent, EPISODES, STEPS, SEED)
        elapsed = time.time() - t0
        results[(mod.TEAM_NAME, mod.ALGO_NAME)] = dict(
            reward=avg_reward, regret=avg_regret, elapsed=elapsed
        )
        print(f"→ {mod.ALGO_NAME} done in {elapsed:.1f}s")

    # --------- PLOTS ----------
    # 1) Smoothed average reward over time
    plt.figure(figsize=(8,5))
    for (team, algo), res in results.items():
        sm = moving_average(res["reward"], SMOOTH_WINDOW)
        plt.plot(sm, label=f"{algo} ({team})")
    plt.xlabel("Step"); plt.ylabel("Avg reward (smoothed)")
    plt.title(f"Average Reward (episodes={EPISODES}, steps={STEPS}, window={SMOOTH_WINDOW})")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) Cumulative regret (lower is better)
    plt.figure(figsize=(8,5))
    for (team, algo), res in results.items():
        plt.plot(res["regret"], label=f"{algo} ({team})")
    plt.xlabel("Step"); plt.ylabel("Average cumulative regret")
    plt.title("Cumulative Regret (lower is better)")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) Summary bar: final avg cumulative regret per agent
    labels = []
    finals = []
    for (team, algo), res in results.items():
        labels.append(f"{algo}\n({team})")
        finals.append(res["regret"][-1])
    order = np.argsort(finals)  # ascending: best first
    plt.figure(figsize=(max(7, 0.7*len(labels)),5))
    plt.bar(range(len(order)), [finals[i] for i in order])
    plt.xticks(range(len(order)), [labels[i] for i in order], rotation=45, ha="right")
    plt.ylabel("Final avg cumulative regret")
    plt.title("Summary: lower is better")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
