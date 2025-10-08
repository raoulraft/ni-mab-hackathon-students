# 🧠 Multi-Armed Bandit Hackathon – Student Edition  
**Course:** Network Intelligence  
**Topic:** Reinforcement Learning Foundations — Bandit Algorithms  

---

## 🎯 Objective

In this mini-project you will:
- Understand the concept of **exploration vs exploitation** through a practical experiment.  
- Implement your own **Multi-Armed Bandit (MAB)** algorithm.  
- Test how well it performs on a simulated environment.  
- Compete (friendly!) with your classmates in a final evaluation.

This is your **first Python project**, so the goal is not just to win — it’s to **learn by doing**.

---

## 🧩 What Is a Multi-Armed Bandit?

Imagine you enter a casino with 10 slot machines (“arms”).  
Each arm gives a reward when you pull it, but the average reward is **different and unknown**.  
Your mission is to find a strategy that maximizes your total reward:
- **Explore** → try different arms to gather information.  
- **Exploit** → focus on the arm that seems best so far.

That’s exactly what you’ll do in code!

---

## 📁 Project Structure

```
mab-hackathon-student/
│
├── README.md                   ← This document (read it carefully!)
│
├── submissions/
│   ├── template_agent.py        ← Template file: COPY and EDIT this!
│   └── my_agent.py              ← Your version (you will create it)
│
└── student_sandbox/
    ├── simple_env.py            ← The environment (do NOT edit)
    └── quick_test.py            ← Testing tool (you will use it to run and plot)
```

---

## 🧰 Required Software

You only need:
- **Python 3.10 or higher**
- **Pip** (Python package manager)

If you don’t have them yet:  
👉 [Download Python](https://www.python.org/downloads/) and tick **“Add Python to PATH”** during installation.

---

## ⚙️ Installing Dependencies

Open a terminal inside the project folder and run:

```bash
pip install numpy matplotlib
```

That’s all you need — these libraries handle math and plotting.

---

## 🧰 Option A — Recommended Setup (Virtual Environment)

To avoid conflicts with other Python packages on your computer, we’ll use a **virtual environment**.  
Run these commands from the project’s root directory (`mab-hackathon-student/`):

```bash
# 1️⃣  Create a virtual environment
python3 -m venv .venv

# 2️⃣  Activate it
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3️⃣  Upgrade pip and install dependencies inside it
python -m pip install --upgrade pip
pip install numpy matplotlib
```

You only need to do this **once**.  
Next time you work on the project:

```bash
source .venv/bin/activate          # macOS / Linux
# or
.venv\Scripts\Activate.ps1         # Windows
```

Then run:

```bash
python student_sandbox/quick_test.py
```

To exit the environment:

```bash
deactivate
```

---

## 🔍 Step 1 — Understand Each File

### 🧠 `simple_env.py`  
This is the **simulation environment** (you don’t touch it).  
It mimics a casino with multiple slot machines (“arms”).  
Each arm has a hidden average reward `q_star[a]`.  
When you call:

```python
reward, info = env.step(arm)
```

you get a **random reward** around that average.

💡 Think of it as a black box that returns rewards when you choose an arm.

---

### ⚙️ `template_agent.py`  
This is the **starting point** of your algorithm.

It defines:
- Your **team name** and **algorithm name**
- The **class structure** that your code must follow
- A **working example**: a simple *ε-greedy* agent.

You will copy this file and edit it.

---

### 📈 `quick_test.py`  
This file lets you **test and visualize** how your algorithm performs.  
You don’t need to edit it, but you’ll run it a lot.

It:
1. Loads your agent automatically.
2. Runs it for many episodes (games).
3. Plots the **average reward curve** so you can see if your algorithm improves over time.

---

## ✏️ Step 2 — Create Your Own Agent

1. Go inside the folder `submissions/`
2. **Copy** the file `template_agent.py`
3. **Rename** the copy as:

```
my_agent.py
```

This is the **only file you will submit**.

---

## 🧠 Step 3 — Fill In Your Agent’s Information

At the top of `my_agent.py`, change:

```python
TEAM_NAME = "TEAM_NAME_HERE"
ALGO_NAME = "MyEpsGreedy"
```

For example:
```python
TEAM_NAME = "Alice_Rossi"
ALGO_NAME = "DecayingEpsilonGreedy"
```

---

## 💻 Step 4 — Understand the Class `BanditAgent`

You’ll see:

```python
class BanditAgent:
    def __init__(self, n_arms: int, seed: int, epsilon: float=0.1):
        ...
    def reset(self, seed: int | None = None):
        ...
    def select_arm(self, context=None) -> int:
        ...
    def update(self, arm: int, reward: float, info: dict | None = None):
        ...
```

Here’s what each function does:

| Function | Called by | You should… |
|:----------|:-----------|:-------------|
| `__init__()` | When the agent is created | Set up variables, e.g. `epsilon`, `Q` values, counters |
| `reset()` | At the start of every episode | Reinitialize everything |
| `select_arm()` | Every step before the environment returns a reward | Choose which arm to play |
| `update()` | After you get a reward | Update your statistics (means, counters, etc.) |

✅ **You are free to change the logic inside these methods.**  
❌ **Don’t change their names or arguments.**

---

## 🔬 Step 5 — Run and Visualize Your Algorithm

Open a terminal and run:

```bash
python student_sandbox/quick_test.py
```

If everything is working, you’ll see a **plot** like this:

```
Average reward increasing over time →
Good! It means your agent is learning.
```

If you see errors:
- Check that your file is called `my_agent.py`
- Check that all function names are correct (`select_arm`, `update`, …)

---

## 🧮 Step 6 — Experiment and Improve

Try to modify your code to implement different algorithms:
- Change the exploration rate `epsilon`
- Make `epsilon` **decrease over time** (`epsilon = 1 / t`)
- Implement **UCB (Upper Confidence Bound)** or **Thompson Sampling**

Each time, re-run:

```bash
python student_sandbox/quick_test.py
```

and compare your results!

---

## 🏁 Step 7 — Submit

When you’re happy with your results:

1. Make sure your file runs correctly.
2. Submit only this file:

```
submissions/my_agent.py
```

No other files are needed.  
The professor will test it on hidden environments.

---

## 🧠 Tips & Good Practices

- Keep your variable names clear, e.g. `self.counts`, `self.Q`.
- Add comments to explain your idea.
- Start simple, then improve gradually.
- You can test multiple agents:  
  copy `my_agent.py` → `my_agent_v2.py` → `my_agent_v3.py` and switch which one runs in `quick_test.py`.

---

## 💬 Need Help?

You can ask ChatGPT for help with coding *as long as* you follow the required structure.

Example prompts you can try:
- “Explain the difference between exploration and exploitation in MAB.”
- “Show me pseudocode for UCB1 using incremental mean updates.”
- “How do I make epsilon decay over time in an epsilon-greedy algorithm?”

---

## 🧾 Summary Table

| File | Purpose | Can I edit it? |
|:--|:--|:--:|
| `submissions/template_agent.py` | Template with working example | ✅ copy & edit |
| `submissions/my_agent.py` | Your implementation | ✅ |
| `student_sandbox/simple_env.py` | The environment | ❌ |
| `student_sandbox/quick_test.py` | Test and visualize | ⚠️ only if you want to change plots |
| `README.md` | Instructions | ✅ (for notes) |

---

## 🧡 Closing Notes

This small project is designed to make you:
- write **your first working AI agent**,
- understand **reinforcement learning** intuitively,
- and gain confidence in experimenting with code.

Be creative, curious, and have fun — your algorithm will soon compete in the class leaderboard!
