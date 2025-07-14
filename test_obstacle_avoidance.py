import numpy as np
import pygame
import os
from environment.env import SimpleSensorFusionEnv
import matplotlib.pyplot as plt
import seaborn as sns

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

def test_obstacle_avoidance(n_bins=4, alpha=0.1, episodes=5):
    """Test obstacle avoidance capabilities with detailed analysis"""
    
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
    obstacle_encounters = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        done = False
        total_reward = 0
        steps = 0
        close_encounters = 0
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1} - OBSTACLE AVOIDANCE DEMONSTRATION")
        print(f"{'='*60}")
        print(f"Initial observation: Distance={obs[0]:.3f}, Angle={obs[1]:.3f}, Type={'Moving' if obs[2] > 0.5 else 'Static'}")
        
        while not done:
            # Choose best action (no exploration during testing)
            action = np.argmax(Q[state[0], state[1], state[2]])
            q_values = Q[state[0], state[1], state[2]]
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretizer.discretize(next_obs)
            
            # Check for close encounters (distance < 0.3)
            if next_obs[0] < 0.3:
                close_encounters += 1
                print(f"⚠️  CLOSE ENCOUNTER #{close_encounters}: Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}")
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render the environment
            env.render()
            
            # Print action taken with Q-values
            action_names = ["LEFT", "RIGHT", "STRAIGHT"]
            print(f"Step {steps:3d}: {action_names[action]:8s} | Q-values: [{q_values[0]:6.3f}, {q_values[1]:6.3f}, {q_values[2]:6.3f}] | "
                  f"Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}, "
                  f"Type={'Moving' if next_obs[2] > 0.5 else 'Static'}")
            
            # Check for pygame quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        total_rewards.append(total_reward)
        obstacle_encounters.append(close_encounters)
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Close encounters: {close_encounters}")
        print(f"  Success: {'✅ REACHED GOAL' if total_reward > 5 else '❌ FAILED'}")
    
    env.close()
    
    print(f"\n{'='*60}")
    print("OBSTACLE AVOIDANCE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Average reward across {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Best episode reward: {np.max(total_rewards):.2f}")
    print(f"Average close encounters per episode: {np.mean(obstacle_encounters):.1f}")
    print(f"Success rate: {sum(1 for r in total_rewards if r > 5)/len(total_rewards)*100:.1f}%")
    
    # Show Q-table analysis for obstacle avoidance
    print(f"\nQ-Table Analysis for Obstacle Avoidance:")
    print(f"  Static obstacles - Best actions by distance/angle:")
    static_q = Q[:, :, 0, :]  # Static obstacles
    for d in range(n_bins):
        for a in range(n_bins):
            best_action = np.argmax(static_q[d, a])
            action_name = ["LEFT", "RIGHT", "STRAIGHT"][best_action]
            print(f"    Distance bin {d}, Angle bin {a}: {action_name}")
    
    print(f"\n  Moving obstacles - Best actions by distance/angle:")
    moving_q = Q[:, :, 1, :]  # Moving obstacles
    for d in range(n_bins):
        for a in range(n_bins):
            best_action = np.argmax(moving_q[d, a])
            action_name = ["LEFT", "RIGHT", "STRAIGHT"][best_action]
            print(f"    Distance bin {d}, Angle bin {a}: {action_name}")

if __name__ == "__main__":
    print("Testing Q-learning agent's obstacle avoidance capabilities...")
    print("This will show detailed analysis of how the agent navigates around obstacles.")
    
    # Test with different configurations to show various obstacle scenarios
    print("\nTesting with 4x4 discretization (coarse but effective):")
    test_obstacle_avoidance(n_bins=4, alpha=0.1, episodes=3)
    
    print("\n" + "="*80)
    print("Testing with 8x8 discretization (finer control):")
    test_obstacle_avoidance(n_bins=8, alpha=0.2, episodes=2)
