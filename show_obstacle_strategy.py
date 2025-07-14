import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_obstacle_strategy(n_bins=4, alpha=0.1):
    """Analyze and visualize the learned obstacle avoidance strategy"""
    
    # Load the trained Q-table
    q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
    if not os.path.exists(q_table_path):
        print(f"Q-table not found at {q_table_path}")
        return
    
    Q = np.load(q_table_path)
    print(f"Loaded Q-table with shape: {Q.shape}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Q-Learning Obstacle Avoidance Strategy (n_bins={n_bins}, alpha={alpha})', fontsize=16)
    
    action_names = ["LEFT", "RIGHT", "STRAIGHT"]
    obstacle_types = ["Static", "Moving"]
    
    # Plot Q-values for each action and obstacle type
    for obs_type in range(2):
        for action in range(3):
            ax = axes[obs_type, action]
            
            # Extract Q-values for this action and obstacle type
            q_values = Q[:, :, obs_type, action]
            
            # Create heatmap
            sns.heatmap(q_values, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                       xticklabels=[f'{i:.2f}' for i in np.linspace(-1, 1, n_bins)],
                       yticklabels=[f'{i:.2f}' for i in np.linspace(0, 1, n_bins)],
                       ax=ax)
            
            ax.set_title(f'{obstacle_types[obs_type]} Obstacles - {action_names[action]}')
            ax.set_xlabel('Angle (normalized)')
            ax.set_ylabel('Distance (normalized)')
    
    plt.tight_layout()
    plt.savefig(f'results/obstacle_strategy_analysis_{n_bins}_{alpha}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print strategy analysis
    print(f"\n{'='*60}")
    print(f"OBSTACLE AVOIDANCE STRATEGY ANALYSIS")
    print(f"{'='*60}")
    
    for obs_type in range(2):
        print(f"\n{obstacle_types[obs_type]} OBSTACLES:")
        print("-" * 40)
        
        # Find best action for each state
        best_actions = np.argmax(Q[:, :, obs_type, :], axis=2)
        
        for d in range(n_bins):
            for a in range(n_bins):
                best_action = best_actions[d, a]
                q_value = Q[d, a, obs_type, best_action]
                
                # Convert bin indices to actual values
                distance_val = (d + 0.5) / n_bins  # Center of bin
                angle_val = (a - n_bins//2 + 0.5) * 2 / n_bins  # Center of bin
                
                print(f"  Distance {distance_val:.2f}, Angle {angle_val:+.2f}: {action_names[best_action]} (Q={q_value:.3f})")
    
    # Analyze patterns
    print(f"\n{'='*60}")
    print("STRATEGY PATTERNS:")
    print(f"{'='*60}")
    
    for obs_type in range(2):
        print(f"\n{obstacle_types[obs_type]} Obstacles:")
        
        # Count actions for different scenarios
        static_q = Q[:, :, obs_type, :]
        
        # Close obstacles (distance bins 0-1)
        close_obstacles = static_q[:2, :, :]
        close_best = np.argmax(close_obstacles, axis=2)
        
        # Far obstacles (distance bins 2-3)
        far_obstacles = static_q[2:, :, :]
        far_best = np.argmax(far_obstacles, axis=2)
        
        print(f"  Close obstacles (distance < 0.5):")
        for action in range(3):
            count = np.sum(close_best == action)
            print(f"    {action_names[action]}: {count} times")
        
        print(f"  Far obstacles (distance > 0.5):")
        for action in range(3):
            count = np.sum(far_best == action)
            print(f"    {action_names[action]}: {count} times")

def compare_strategies():
    """Compare strategies across different configurations"""
    
    configs = [
        (4, 0.1),
        (4, 0.2), 
        (8, 0.1),
        (8, 0.2)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Obstacle Avoidance Strategy Comparison', fontsize=16)
    
    for idx, (n_bins, alpha) in enumerate(configs):
        q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
        if not os.path.exists(q_table_path):
            continue
            
        Q = np.load(q_table_path)
        ax = axes[idx//2, idx%2]
        
        # Show best action for static obstacles
        best_actions = np.argmax(Q[:, :, 0, :], axis=2)
        
        # Create custom colormap for actions
        colors = ['red', 'blue', 'green']  # LEFT, RIGHT, STRAIGHT
        cmap = plt.cm.colors.ListedColormap(colors)
        
        im = ax.imshow(best_actions, cmap=cmap, aspect='auto')
        ax.set_title(f'n_bins={n_bins}, alpha={alpha}')
        ax.set_xlabel('Angle bins')
        ax.set_ylabel('Distance bins')
        
        # Add text annotations
        for i in range(n_bins):
            for j in range(n_bins):
                action = best_actions[i, j]
                action_name = ["L", "R", "S"][action]
                ax.text(j, i, action_name, ha='center', va='center', 
                       color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Analyzing Q-learning obstacle avoidance strategies...")
    
    # Analyze individual strategies
    print("\nAnalyzing 4x4 discretization strategy:")
    analyze_obstacle_strategy(n_bins=4, alpha=0.1)
    
    print("\nAnalyzing 8x8 discretization strategy:")
    analyze_obstacle_strategy(n_bins=8, alpha=0.2)
    
    # Compare strategies
    print("\nComparing strategies across configurations:")
    compare_strategies()
