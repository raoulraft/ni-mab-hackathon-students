import importlib.util, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from simple_env import SimpleBandit

def load_student(module_path):
    spec = importlib.util.spec_from_file_location("student_agent", module_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore
    return mod

def run(agent_factory, episodes=20, steps=1000, seed=7):
    avg_r = np.zeros(steps)
    for ep in range(episodes):
        env = SimpleBandit(n_arms=10, seed=seed+ep)
        agent = agent_factory(env.n_arms, seed+ep)
        agent.reset(seed+ep)
        for t in range(steps):
            a = agent.select_arm()
            r, info = env.step(a)
            agent.update(a, r, info)
            avg_r[t] += r
    avg_r /= episodes
    return avg_r

if __name__ == "__main__":
    here = os.path.dirname(__file__)
    my_agent = os.path.join(here, "..", "submissions", "my_agent.py")
    template = os.path.join(here, "..", "submissions", "template_agent.py")

    agent_path = my_agent if os.path.exists(my_agent) else template
    if agent_path == template:
        print("[INFO] No my_agent.py found — using template_agent.py instead.")
        print("       Create it with: cp submissions/template_agent.py submissions/my_agent.py")

    mod = load_student(agent_path)
    rewards = run(mod.create_agent, episodes=20, steps=1000, seed=7)

    plt.figure(figsize=(7,4.5))
    plt.plot(rewards, label=f"{mod.ALGO_NAME}")
    plt.xlabel("Step"); plt.ylabel("Average reward"); plt.title("Student sandbox — average reward")
    plt.legend(); plt.grid(True, alpha=0.3); plt.show()
