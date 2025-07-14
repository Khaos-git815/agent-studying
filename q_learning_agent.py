import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from environment.env import SimpleSensorFusionEnv
from mpl_toolkits.mplot3d import Axes3D

# Discretization helper
class Discretizer:
    def __init__(self, bins):
        self.bins = bins  # List of arrays for each dimension
    def discretize(self, obs):
        idxs = [int(np.digitize(o, b) - 1) for o, b in zip(obs, self.bins)]
        # Clip indices to valid range
        idxs[0] = np.clip(idxs[0], 0, n_bins-1)  # distance
        idxs[1] = np.clip(idxs[1], 0, n_bins-1)  # angle
        idxs[2] = np.clip(idxs[2], 0, 1)         # type_confidence
        return tuple(idxs)

# Initialize environment
env = SimpleSensorFusionEnv()

# --- Hyperparameter sweep settings ---
# You can add/remove values to try more/less combos
n_bins_list = [4, 8]
alpha_list = [0.05, 0.1, 0.2]
gamma = 0.95
epsilon = 1.0
decay = 0.995
min_epsilon = 0.01
episodes = 2000

# --- Recommended combos for this environment ---
# (n_bins, alpha)
# (4, 0.1), (8, 0.05), (8, 0.2)

import shutil

for n_bins in n_bins_list:
    for alpha in alpha_list:
        # Set up discretization bins
        obs_bins = [
            np.linspace(0, 1, n_bins+1),   # distance
            np.linspace(-1, 1, n_bins+1),  # angle
            np.linspace(0, 1, 2)           # type_confidence (static/moving)
        ]
        discretizer = Discretizer(obs_bins)
        n_actions = env.action_space.n
        Q = np.zeros((n_bins, n_bins, 2, n_actions))
        rewards = []
        eps = epsilon
        for episode in range(episodes):
            obs, _ = env.reset()
            state = discretizer.discretize(obs)
            done = False
            total_reward = 0
            while not done:
                if np.random.rand() < eps:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[state[0], state[1], state[2]])
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = discretizer.discretize(next_obs)
                Q[state[0], state[1], state[2], action] += alpha * (
                    reward + gamma * np.max(Q[next_state[0], next_state[1], next_state[2]]) - Q[state[0], state[1], state[2], action]
                )
                state = next_state
                total_reward += reward
            eps = max(min_epsilon, eps * decay)
            rewards.append(total_reward)
        # Save results in a subfolder
        result_dir = f"results/bins{n_bins}_alpha{alpha}"
        os.makedirs(result_dir, exist_ok=True)
        np.save(f"{result_dir}/q_table.npy", Q)
        # Plot Q-table heatmap for static
        plt.figure(figsize=(10, 7))
        sns.heatmap(Q[:, :, 0, :].max(axis=2), annot=False, cmap="viridis")
        plt.title(f"Q-table Heatmap (static) n_bins={n_bins}, alpha={alpha}")
        plt.xlabel("Angle bins")
        plt.ylabel("Distance bins")
        plt.savefig(f"{result_dir}/q_table_plot.png")
        plt.close()
        # Plot Q-table heatmap for moving
        plt.figure(figsize=(10, 7))
        sns.heatmap(Q[:, :, 1, :].max(axis=2), annot=False, cmap="viridis")
        plt.title(f"Q-table Heatmap (moving) n_bins={n_bins}, alpha={alpha}")
        plt.xlabel("Angle bins")
        plt.ylabel("Distance bins")
        plt.savefig(f"{result_dir}/q_table_plot_moving.png")
        plt.close()
        # 3D Q-table surface plot for a selected action (e.g., action=0) and type_conf=0 (static)
        action_idx = 0  # You can change this to plot for other actions
        X, Y = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
        Z_static = Q[:, :, 0, action_idx]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_static, cmap='viridis')
        ax.set_title(f'Q-table 3D Surface (static, action={action_idx}) n_bins={n_bins}, alpha={alpha}')
        ax.set_xlabel('Angle bins')
        ax.set_ylabel('Distance bins')
        ax.set_zlabel('Q-value')
        plt.savefig(f"{result_dir}/q_table_surface_static.png")
        plt.close()
        # 3D Q-table surface plot for a selected action (e.g., action=0) and type_conf=1 (moving)
        Z_moving = Q[:, :, 1, action_idx]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z_moving, cmap='viridis')
        ax.set_title(f'Q-table 3D Surface (moving, action={action_idx}) n_bins={n_bins}, alpha={alpha}')
        ax.set_xlabel('Angle bins')
        ax.set_ylabel('Distance bins')
        ax.set_zlabel('Q-value')
        plt.savefig(f"{result_dir}/q_table_surface_moving.png")
        plt.close()
        # Plot rewards with moving average overlay
        window = 100
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.figure(figsize=(10, 5))
            plt.plot(rewards, label='Reward per Episode', alpha=0.5)
            plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Avg (window={window})', color='red')
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Reward per Episode (n_bins={n_bins}, alpha={alpha})")
            plt.legend()
            plt.savefig(f"{result_dir}/reward_plot.png")
            plt.close()
        else:
            plt.figure(figsize=(10, 5))
            plt.plot(rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title(f"Reward per Episode (n_bins={n_bins}, alpha={alpha})")
            plt.savefig(f"{result_dir}/reward_plot.png")
            plt.close()
        print(f"Finished: n_bins={n_bins}, alpha={alpha} -> results in {result_dir}")

# --- End of sweep ---
print("\nTraining completed! All configurations have been trained and results saved.")
print("To test a trained agent, run: python test_trained_agent.py") 