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

def extreme_obstacle_test(n_bins=4, alpha=0.1, episodes=3):
    """Test agent in extreme obstacle scenarios with detailed analysis"""
    
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
    navigation_stats = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = discretizer.discretize(obs)
        done = False
        total_reward = 0
        steps = 0
        close_encounters = 0
        action_counts = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0}
        obstacle_switches = 0
        last_obstacle_type = None
        
        print(f"\n{'='*70}")
        print(f"EXTREME OBSTACLE TEST - EPISODE {episode + 1}")
        print(f"{'='*70}")
        print(f"Initial observation: Distance={obs[0]:.3f}, Angle={obs[1]:.3f}, Type={'Moving' if obs[2] > 0.5 else 'Static'}")
        
        while not done:
            # Choose best action (no exploration during testing)
            action = np.argmax(Q[state[0], state[1], state[2]])
            q_values = Q[state[0], state[1], state[2]]
            
            # Take action
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretizer.discretize(next_obs)
            
            # Track obstacle type switches
            current_obstacle_type = 'Moving' if next_obs[2] > 0.5 else 'Static'
            if last_obstacle_type and last_obstacle_type != current_obstacle_type:
                obstacle_switches += 1
                print(f"🔄 OBSTACLE SWITCH: {last_obstacle_type} → {current_obstacle_type}")
            last_obstacle_type = current_obstacle_type
            
            # Check for close encounters (distance < 0.3)
            if next_obs[0] < 0.3:
                close_encounters += 1
                print(f"⚠️  CLOSE ENCOUNTER #{close_encounters}: Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}")
            
            # Check for very close encounters (distance < 0.2)
            if next_obs[0] < 0.2:
                print(f"🚨 CRITICAL ENCOUNTER: Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}")
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Count actions
            action_names = ["LEFT", "RIGHT", "STRAIGHT"]
            action_counts[action_names[action]] += 1
            
            # Render the environment
            env.render()
            
            # Print detailed action analysis
            print(f"Step {steps:3d}: {action_names[action]:8s} | Q-values: [{q_values[0]:6.3f}, {q_values[1]:6.3f}, {q_values[2]:6.3f}] | "
                  f"Distance={next_obs[0]:.3f}, Angle={next_obs[1]:.3f}, "
                  f"Type={current_obstacle_type}")
            
            # Check for pygame quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        total_rewards.append(total_reward)
        
        # Calculate navigation statistics
        nav_stats = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'steps': steps,
            'close_encounters': close_encounters,
            'critical_encounters': sum(1 for i in range(steps) if i < 0.2),
            'obstacle_switches': obstacle_switches,
            'action_distribution': action_counts,
            'success': total_reward > 5
        }
        navigation_stats.append(nav_stats)
        
        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps taken: {steps}")
        print(f"  Close encounters: {close_encounters}")
        print(f"  Obstacle type switches: {obstacle_switches}")
        print(f"  Action distribution: {action_counts}")
        print(f"  Success: {'✅ REACHED GOAL' if total_reward > 5 else '❌ FAILED'}")
    
    env.close()
    
    # Print comprehensive analysis
    print(f"\n{'='*70}")
    print("EXTREME OBSTACLE TEST ANALYSIS")
    print(f"{'='*70}")
    print(f"Average reward across {episodes} episodes: {np.mean(total_rewards):.2f}")
    print(f"Best episode reward: {np.max(total_rewards):.2f}")
    print(f"Success rate: {sum(1 for r in total_rewards if r > 5)/len(total_rewards)*100:.1f}%")
    
    # Action analysis
    total_actions = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0}
    for stats in navigation_stats:
        for action, count in stats['action_distribution'].items():
            total_actions[action] += count
    
    print(f"\nOverall Action Distribution:")
    total_steps = sum(total_actions.values())
    for action, count in total_actions.items():
        percentage = (count / total_steps) * 100
        print(f"  {action}: {count} times ({percentage:.1f}%)")
    
    # Obstacle handling analysis
    total_switches = sum(stats['obstacle_switches'] for stats in navigation_stats)
    total_close = sum(stats['close_encounters'] for stats in navigation_stats)
    print(f"\nObstacle Handling Statistics:")
    print(f"  Total obstacle type switches: {total_switches}")
    print(f"  Total close encounters: {total_close}")
    print(f"  Average switches per episode: {total_switches/episodes:.1f}")
    print(f"  Average close encounters per episode: {total_close/episodes:.1f}")
    
    # Create visualization
    create_navigation_visualization(navigation_stats, n_bins, alpha)

def create_navigation_visualization(navigation_stats, n_bins, alpha):
    """Create detailed visualization of navigation patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Extreme Obstacle Navigation Analysis (n_bins={n_bins}, alpha={alpha})', fontsize=16)
    
    # Episode performance
    episodes = [stats['episode'] for stats in navigation_stats]
    rewards = [stats['total_reward'] for stats in navigation_stats]
    close_encounters = [stats['close_encounters'] for stats in navigation_stats]
    
    ax1 = axes[0, 0]
    ax1.bar(episodes, rewards, color=['green' if r > 5 else 'red' for r in rewards])
    ax1.set_title('Episode Performance')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.axhline(y=5, color='orange', linestyle='--', label='Success Threshold')
    ax1.legend()
    
    # Close encounters
    ax2 = axes[0, 1]
    ax2.bar(episodes, close_encounters, color='orange')
    ax2.set_title('Close Encounters per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Number of Close Encounters')
    
    # Action distribution
    ax3 = axes[1, 0]
    actions = ['LEFT', 'RIGHT', 'STRAIGHT']
    total_actions = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0}
    for stats in navigation_stats:
        for action, count in stats['action_distribution'].items():
            total_actions[action] += count
    
    action_counts = [total_actions[action] for action in actions]
    colors = ['red', 'blue', 'green']
    ax3.pie(action_counts, labels=actions, colors=colors, autopct='%1.1f%%')
    ax3.set_title('Overall Action Distribution')
    
    # Obstacle switches
    ax4 = axes[1, 1]
    switches = [stats['obstacle_switches'] for stats in navigation_stats]
    ax4.bar(episodes, switches, color='purple')
    ax4.set_title('Obstacle Type Switches per Episode')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Switches')
    
    plt.tight_layout()
    plt.savefig(f'results/extreme_obstacle_analysis_{n_bins}_{alpha}.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_different_scenarios():
    """Test different challenging scenarios"""
    
    scenarios = [
        {"name": "High Precision (8x8)", "n_bins": 8, "alpha": 0.2},
        {"name": "Balanced (4x4)", "n_bins": 4, "alpha": 0.1},
        {"name": "Fast Learning (4x4)", "n_bins": 4, "alpha": 0.2}
    ]
    
    print("Testing different Q-learning configurations in extreme scenarios...")
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Testing: {scenario['name']}")
        print(f"{'='*50}")
        extreme_obstacle_test(scenario['n_bins'], scenario['alpha'], episodes=2)

if __name__ == "__main__":
    print("Running Extreme Obstacle Avoidance Tests...")
    print("This will test the agent in challenging scenarios with detailed analysis.")
    
    # Test different scenarios
    test_different_scenarios() 