import numpy as np
import pygame
import os
from environment.env import SimpleSensorFusionEnv
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from PIL import Image

# Discretization helper (same as in training)
class Discretizer:
    def __init__(self, bins):
        self.bins = bins
    def discretize(self, obs):
        idxs = [int(np.digitize(o, b) - 1) for o, b in zip(obs, self.bins)]
        idxs[0] = np.clip(idxs[0], 0, len(self.bins[0])-2)  # distance
        idxs[1] = np.clip(idxs[1], 0, len(self.bins[1])-2)  # angle
        idxs[2] = np.clip(idxs[2], 0, 1)                    # type_confidence
        return tuple(idxs)

def test_agent(n_bins=4, alpha=0.1, episodes=3, make_gif=False, gif_path=None):
    """Test a trained Q-learning agent in the sensor fusion environment"""
    
    # Load the trained Q-table
    q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
    if not os.path.exists(q_table_path):
        print(f"Q-table not found at {q_table_path}")
        return
    
    Q = np.load(q_table_path)
    print(f"Loaded Q-table with shape: {Q.shape}")
    
    # Set up discretization (same as training)
    obs_bins = [
        np.linspace(0, 1, n_bins+1),   # distance
        np.linspace(-1, 1, n_bins+1),  # angle
        np.linspace(0, 1, 2)           # type_confidence
    ]
    discretizer = Discretizer(obs_bins)
    
    # Initialize environment
    env = SimpleSensorFusionEnv()
    
    total_rewards = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        done = False
        total_reward = 0
        steps = 0
        frames = [] if make_gif and episode == 0 else None
        while not done:
            # Choose best action (no exploration during testing)
            action = np.argmax(Q[state[0], state[1], state[2]])
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretizer.discretize(next_obs)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            env.render()
            if frames is not None:
                # Capture frame from pygame display
                frame = pygame.surfarray.array3d(pygame.display.get_surface())
                frame = np.transpose(frame, (1, 0, 2))
                frames.append(Image.fromarray(frame))
            
            # Print action taken
            action_names = ["LEFT", "RIGHT", "STRAIGHT"]
            print(f"Step {steps}: Action={action_names[action]}, Reward={reward:.2f}, "
                  f"Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}, "
                  f"Type={'Moving' if next_obs[2] > 0.5 else 'Static'}")
            
            # Check for pygame quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1} finished: Total reward = {total_reward:.2f}, Steps = {steps}")
    
    env.close()
    print(f"\nTesting completed!")
    print(f"Average reward across {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Best episode reward: {np.max(total_rewards):.2f}")

    # Plot Q-table heatmap for static obstacles
    plt.figure(figsize=(10, 7))
    sns.heatmap(Q[:, :, 0, :].max(axis=2), annot=False, cmap="viridis")
    plt.title(f"Q-table Heatmap (static) n_bins={n_bins}, alpha={alpha}")
    plt.savefig(f"results/bins{n_bins}_alpha{alpha}/q_table_plot.png")
    plt.show()

    # Plot Q-table heatmap for moving obstacles  
    plt.figure(figsize=(10, 7))
    sns.heatmap(Q[:, :, 1, :].max(axis=2), annot=False, cmap="viridis")
    plt.title(f"Q-table Heatmap (moving) n_bins={n_bins}, alpha={alpha}")
    plt.savefig(f"results/bins{n_bins}_alpha{alpha}/q_table_plot_moving.png")
    plt.show()

    # 3D surface plot for static obstacles
    action_idx = 0  # Plot for action 0 (LEFT)
    X, Y = np.meshgrid(np.arange(n_bins), np.arange(n_bins))
    Z_static = Q[:, :, 0, action_idx]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z_static, cmap='viridis')
    ax.set_title(f'Q-table 3D Surface (static, action={action_idx}) n_bins={n_bins}, alpha={alpha}')
    ax.set_xlabel('Angle bins')
    ax.set_ylabel('Distance bins')
    ax.set_zlabel('Q-value')
    plt.savefig(f"results/bins{n_bins}_alpha{alpha}/q_table_surface_static.png")
    plt.show()

    # Plot rewards
    plt.figure(figsize=(8, 4))
    plt.plot(total_rewards, label='Reward per Episode')
    window = 100
    if len(total_rewards) >= window:
        moving_avg = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(total_rewards)), moving_avg, label=f'Moving Avg (window={window})', color='red')
    else:
        plt.title(f'Reward per Episode (moving average not shown, need >= {window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/bins{n_bins}_alpha{alpha}/reward_plot.png")
    plt.show()

if __name__ == "__main__":
    print("Testing trained Q-learning agent in sensor fusion environment...")
    print("Available configurations:")
    print("1. n_bins=4, alpha=0.05")
    print("2. n_bins=4, alpha=0.1")
    print("3. n_bins=4, alpha=0.2")
    print("4. n_bins=8, alpha=0.05")
    print("5. n_bins=8, alpha=0.1")
    print("6. n_bins=8, alpha=0.2")
    
    # Test with a good configuration
    test_agent(n_bins=4, alpha=0.1, episodes=3, make_gif=True, gif_path="results/agent_demo.gif") 