import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

slippery = False

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)

P = env.unwrapped.P #transition

# actions: 0=Left, 1=Down, 2=Right, 3=Up
A = [0, 1, 2, 3]
A_dict = {0:"left", 1:"down", 2:"right", 3:"up"}
nA = env.action_space.n

# 4x4 grid 
S = [0, 1, 2, 3, 
     4, 5, 6, 7, 
     8, 9, 10, 11, 
     12, 13, 14, 15]
nS = env.observation_space.n

state, info = env.reset()

def mc_control(b_0=None, gamma=0.95, epsilon_s=0.3, max_iter=500, episodes=1000, eps_decay=0.99):
    Q = np.random.uniform(size=(nS, nA))
    visit_counts = np.zeros((nS, nA))
    episode_returns, step_counts, avg_returns = [], [], []
    total_steps = 0
    eps = epsilon_s

    for ep in range(episodes):
        state, _ = env.reset()
        if b_0 is not None:
            env.unwrapped.s = int(b_0)
            state = int(b_0)

        traj_states, traj_actions, traj_rewards = [], [], []

        for _ in range(max_iter):
            total_steps += 1
            if np.random.random() < eps:
                action = np.random.choice(nA)
            else:
                action = np.argmax(Q[state])

            traj_states.append(state)
            traj_actions.append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            traj_rewards.append(reward)
            state = next_state

            if terminated or truncated:
                break

        G = 0.0
        seen = set()
        for idx in reversed(range(len(traj_states))):
            G = traj_rewards[idx] + gamma * G
            key = (traj_states[idx], traj_actions[idx])
            if key not in seen:
                seen.add(key)
                visit_counts[key] += 1
                lr = 0.2
                Q[key] += lr * (G - Q[key])

        total_steps += len(traj_states)
        step_counts.append(total_steps)
        episode_returns.append(G)
        avg_returns.append(np.mean(episode_returns))
        eps *= eps_decay

    policy = np.argmax(Q, axis=1)
    return Q, policy, avg_returns, step_counts


def SARSA(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.3, max_iter=500):
    Q = np.random.uniform(size=(nS, nA))
    eps = epsilon_s
    returns, steps, avg_returns = [], [], []
    total_steps = 0

    for ep in range(1000):
        state, _ = env.reset()
        env.unwrapped.s = b_0
        state = b_0

        if np.random.random() < eps:
            action = np.random.choice(nA)
        else:
            action = np.argmax(Q[state])
        eps *= 0.9

        episode_rewards = []

        for t in range(max_iter):
            total_steps += 1
            next_state, reward, done, _, _ = env.step(action)
            episode_rewards.append(reward)

            if np.random.random() < eps:
                next_action = np.random.choice(nA)
            else:
                next_action = np.argmax(Q[next_state])
            eps *= 0.9

            if done:
                Q[state, action] += alpha * (reward - Q[state, action])
                break
            else:
                Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])

            state, action = next_state, next_action

        G = 0.0
        for r in reversed(episode_rewards):
            G = r + gamma * G
        returns.append(G)
        avg_returns.append(np.mean(returns))
        steps.append(total_steps)

    policy = np.argmax(Q, axis=1)
    return Q, policy, avg_returns, steps


def Q_learning(b_0, gamma=0.95, alpha=0.2, epsilon_s=0.3, max_iter=500):
    Q = np.random.uniform(size=(nS, nA))
    eps = epsilon_s
    returns, steps, avg_returns = [], [], []
    total_steps = 0

    for ep in range(1000):
        state, _ = env.reset()
        env.unwrapped.s = b_0
        state = b_0

        if np.random.random() < eps:
            action = np.random.choice(nA)
        else:
            action = np.argmax(Q[state])
        eps *= 0.9

        episode_rewards = []

        for t in range(max_iter):
            total_steps += 1
            next_state, reward, done, _, _ = env.step(action)
            episode_rewards.append(reward)

            if np.random.random() < eps:
                next_action = np.random.choice(nA)
            else:
                next_action = np.argmax(Q[next_state])
            eps *= 0.9

            if done:
                Q[state, action] += alpha * (reward - Q[state, action])
                break
            else:
                Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state, action = next_state, next_action

        G = 0.0
        for r in reversed(episode_rewards):
            G = r + gamma * G
        returns.append(G)
        avg_returns.append(np.mean(returns))
        steps.append(total_steps)

    policy = np.argmax(Q, axis=1)
    return Q, policy, avg_returns, steps



# main - slippery default false
print(f"SLIPPERY: {slippery}")

Q_MC, pi_MC, r_MC, t_MC  = mc_control(0)
Q_SARSA, pi_SARSA, r_SARSA, t_SARSA = SARSA(0)
Q_Q, pi_Q, r_Q, t_Q = Q_learning(0)

print("MC Policy:", pi_MC,"\n")
print("Q Policy:", pi_Q,"\n")
print("SARSA Policy:", pi_SARSA,"\n")

print("MC Value Function:", "\n", Q_MC, "\n")
print("SARSA Value Function:", "\n", Q_SARSA, "\n")
print("Q Value Function:", "\n", Q_Q, "\n")

plt.figure(figsize=(7,4))
plt.plot(t_MC, r_MC, label='MC Control')
plt.plot(t_SARSA, r_SARSA, label='SARSA')
plt.plot(t_Q, r_Q, label='Q-Learning')
plt.title(f'FrozenLake 4x4 (slippery={slippery}): Learning Curves')
plt.xlabel('Time steps')
plt.ylabel('Mean return (running average)')
plt.legend(loc='best', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

### Slippery = True ###

slippery = True
print(f"SLIPPERY: {slippery}")

Q_MC, pi_MC, r_MC, t_MC  = mc_control(0)
Q_SARSA, pi_SARSA, r_SARSA, t_SARSA = SARSA(0)
Q_Q, pi_Q, r_Q, t_Q = Q_learning(0)

print("MC Policy:", pi_MC,"\n")
print("Q Policy:", pi_Q,"\n")
print("SARSA Policy:", pi_SARSA,"\n")

print("MC Value Function:", "\n", Q_MC, "\n")
print("SARSA Value Function:", "\n", Q_SARSA, "\n")
print("Q Value Function:", "\n", Q_Q, "\n")


plt.figure(figsize=(7,4))
plt.plot(t_MC, r_MC, label='MC Control')
plt.plot(t_SARSA, r_SARSA, label='SARSA')
plt.plot(t_Q, r_Q, label='Q-Learning')
plt.title(f'FrozenLake 4x4 (slippery={slippery}): Learning Curves')
plt.xlabel('Time steps')
plt.ylabel('Mean return (running average)')
plt.legend(loc='best', frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()