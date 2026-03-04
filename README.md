[Potential_Based_Reward_Shaping_in_Deep_Reinforcement_Learning__A_Comparative_Study_on_FrozenLake_Navigation.pdf](https://github.com/user-attachments/files/25751896/Potential_Based_Reward_Shaping_in_Deep_Reinforcement_Learning__A_Comparative_Study_on_FrozenLake_Navigation.pdf)


[README (1).md](https://github.com/user-attachments/files/25751866/README.1.md)
# Potential-Based Reward Shaping in Deep Reinforcement Learning
### A Comparative Study on FrozenLake Navigation

**Authors:** Sundar Ponnusamy, Yundan Zhang, Tarun Malepati, Hannah Teng

---

## Overview

This project presents a comprehensive empirical study investigating the differential impact of **Potential-Based Reward Shaping (PBRS)** across five canonical reinforcement learning algorithms on the FrozenLake gridworld environment. Our results challenge conventional assumptions — sparse rewards outperformed PBRS-shaped rewards in **75% of algorithm-condition combinations**.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Deep_SARSA_for_frozen_lake_finished.ipynb` | Full implementation of Deep SARSA with all experimental conditions |
| `Potential_Based_Reward_Shaping_...pdf` | Full paper with results, analysis, and discussion |

---

## Algorithms Studied

| Algorithm | Type | Key Characteristic |
|-----------|------|--------------------|
| Tabular Q-Learning | Value-Based | Exact Q-table, ε-greedy exploration |
| Deep Q-Network (DQN) | Value-Based | Off-policy, experience replay, target network |
| Deep SARSA | Value-Based | On-policy, conservative updates |
| REINFORCE | Policy-Based | Monte Carlo returns, high variance |
| PPO | Policy-Based | Clipped surrogate objective, actor-critic |

---

## Experimental Setup

- **Environment:** FrozenLake-v1 (Gymnasium)
- **Map Sizes:** 8×8 (64 states), 16×16 (256 states)
- **Dynamics:** Slippery (stochastic, 1/3 directional probability) and Deterministic
- **Reward Structures:** Sparse vs. Potential-Based Shaped
- **Total Conditions:** 40 (5 algorithms × 2 map sizes × 2 dynamics × 2 reward structures)
- **Training:** 10,000 episodes per condition, random seed 42

### Reward Shaping

The shaped reward is defined as:

```
r_shaped(s, a, s') = r(s, a, s') + γΦ(s') − Φ(s)
```

where `Φ(s) = −d_Manhattan(s, goal)` — guiding the agent toward the goal while theoretically preserving the optimal policy.

---

## Key Findings

### 1. Sparse Rewards Beat Shaped Rewards (75% of conditions)
Despite PBRS's theoretical guarantees, dense shaped rewards consistently hurt performance when combined with neural network function approximation. Shaped rewards dilute credit assignment and introduce local optima where agents achieve small consistent rewards without completing episodes.

### 2. Deep SARSA Outperforms DQN by 30–40% in Complex Environments

| Condition | DQN | Deep SARSA |
|-----------|-----|------------|
| 8×8 Slippery (sparse) | 55–75% | 60–70% |
| 16×16 Slippery (sparse) | 50–65% | **95–100%** |
| 16×16 Deterministic | 25–90% | **95–100%** |

DQN's maximization bias compounds with environmental uncertainty. Deep SARSA's on-policy updates account for exploration risk, giving it superior stability in stochastic environments.

### 3. Policy-Based Methods Fail Catastrophically with Shaped Rewards

| Condition | Sparse | Shaped |
|-----------|--------|--------|
| REINFORCE 8×8 Deterministic | **100%** | 0% |
| PPO 8×8 Deterministic | **100%** | 65% |

Dense rewards flatten the return landscape, making it difficult to distinguish successful from unsuccessful trajectories — especially in deterministic environments with limited exploration.

### 4. Overall Algorithm Rankings (averaged across conditions)

1. 🥇 **Deep SARSA** — Consistent 60–100%, minimal variance
2. 🥈 **PPO** — 65–100%, robust to most conditions
3. 🥉 **REINFORCE** — Bimodal (0% or 95–100%)
4. **DQN** — High variance (10–90%)
5. **Tabular Q-Learning** — Frequent catastrophic failures

---

## Why PBRS Failed in Deep RL

The original PBRS theory assumes:
- Exact value representation
- Exhaustive state-space exploration
- Convergence guarantees

All three assumptions are violated in deep RL. Neural function approximation introduces generalization error, finite training limits exploration coverage, and experience replay de-correlates temporal structure in ways that interact poorly with dense reward signals.

---

## How to Run

Open `Deep_SARSA_for_frozen_lake_finished.ipynb` in Jupyter or Google Colab.

**Dependencies:**
```bash
pip install gymnasium numpy torch matplotlib
```

---

## Citation

If you use this work, please cite:

S. Ponnusamy, Y. Zhang, T. Malepati, and H. Teng, "Potential-Based Reward Shaping in
Deep Reinforcement Learning: A Comparative Study on FrozenLake Navigation," 2024.

---

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, MIT Press, 2018
- Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017
- Ng, Harada & Russell, "Policy invariance under reward transformations," ICML 1999
- Zhao et al., "Deep reinforcement learning with experience replay based on SARSA," IEEE SSCI, 2016

