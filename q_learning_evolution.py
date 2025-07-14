import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from environment.env import SimpleSensorFusionEnv

def analyze_q_table_evolution(n_bins=4, alpha=0.1):
    """Analyze how Q-values evolve and show learning patterns"""
    
    # Load the trained Q-table
    q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
    if not os.path.exists(q_table_path):
        print(f"Q-table not found at {q_table_path}")
        return
    
    Q = np.load(q_table_path)
    print(f"Loaded Q-table with shape: {Q.shape}")
    
    # Create comprehensive Q-table analysis
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Q-Table Evolution Analysis (n_bins={n_bins}, alpha={alpha})', fontsize=16)
    
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
            
            ax.set_title(f'{obstacle_types[obs_type]} - {action_names[action]}')
            ax.set_xlabel('Angle (normalized)')
            ax.set_ylabel('Distance (normalized)')
    
    # Add Q-value distribution analysis
    ax_dist = axes[2, 0]
    all_q_values = Q.flatten()
    ax_dist.hist(all_q_values, bins=50, alpha=0.7, color='blue')
    ax_dist.set_title('Q-Value Distribution')
    ax_dist.set_xlabel('Q-Value')
    ax_dist.set_ylabel('Frequency')
    ax_dist.axvline(x=np.mean(all_q_values), color='red', linestyle='--', label=f'Mean: {np.mean(all_q_values):.3f}')
    ax_dist.legend()
    
    # Add learning confidence analysis
    ax_conf = axes[2, 1]
    max_q_values = np.max(Q, axis=3)
    confidence = max_q_values - np.mean(Q, axis=3)
    sns.heatmap(confidence.mean(axis=2), annot=True, fmt='.2f', cmap='viridis', ax=ax_conf)
    ax_conf.set_title('Learning Confidence (Max - Mean Q)')
    ax_conf.set_xlabel('Angle bins')
    ax_conf.set_ylabel('Distance bins')
    
    # Add action preference analysis
    ax_pref = axes[2, 2]
    best_actions = np.argmax(Q, axis=3)
    action_preferences = np.zeros((n_bins, n_bins, 3))
    for i in range(n_bins):
        for j in range(n_bins):
            for k in range(2):  # obstacle types
                action = best_actions[i, j, k]
                action_preferences[i, j, action] += 1
    
    # Show most preferred action for each state
    most_preferred = np.argmax(action_preferences, axis=2)
    colors = ['red', 'blue', 'green']
    cmap = plt.cm.colors.ListedColormap(colors)
    im = ax_pref.imshow(most_preferred, cmap=cmap, aspect='auto')
    ax_pref.set_title('Most Preferred Actions')
    ax_pref.set_xlabel('Angle bins')
    ax_pref.set_ylabel('Distance bins')
    
    # Add text annotations
    for i in range(n_bins):
        for j in range(n_bins):
            action = most_preferred[i, j]
            action_name = ["L", "R", "S"][action]
            ax_pref.text(j, i, action_name, ha='center', va='center', 
                        color='white', fontweight='bold')
    
    # Add Q-value statistics
    ax_stats = axes[2, 3]
    metrics = ['Mean', 'Std', 'Min', 'Max', 'Positive %', 'Negative %']
    values = [
        f"{np.mean(all_q_values):.3f}",
        f"{np.std(all_q_values):.3f}",
        f"{np.min(all_q_values):.3f}",
        f"{np.max(all_q_values):.3f}",
        f"{(all_q_values > 0).sum() / len(all_q_values) * 100:.1f}%",
        f"{(all_q_values < 0).sum() / len(all_q_values) * 100:.1f}%"
    ]
    
    ax_stats.axis('tight')
    ax_stats.axis('off')
    table = ax_stats.table(cellText=[[m, v] for m, v in zip(metrics, values)], 
                          colLabels=['Metric', 'Value'],
                          cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    ax_stats.set_title('Q-Table Statistics')
    
    plt.tight_layout()
    plt.savefig(f'results/q_table_evolution_{n_bins}_{alpha}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print(f"\n{'='*60}")
    print(f"Q-TABLE EVOLUTION ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nQ-Value Statistics:")
    print(f"  Mean Q-value: {np.mean(all_q_values):.3f}")
    print(f"  Standard deviation: {np.std(all_q_values):.3f}")
    print(f"  Minimum Q-value: {np.min(all_q_values):.3f}")
    print(f"  Maximum Q-value: {np.max(all_q_values):.3f}")
    print(f"  Positive Q-values: {(all_q_values > 0).sum()} / {len(all_q_values)} ({(all_q_values > 0).sum() / len(all_q_values) * 100:.1f}%)")
    print(f"  Negative Q-values: {(all_q_values < 0).sum()} / {len(all_q_values)} ({(all_q_values < 0).sum() / len(all_q_values) * 100:.1f}%)")
    
    # Analyze learning patterns
    print(f"\nLearning Pattern Analysis:")
    
    for obs_type in range(2):
        print(f"\n{obstacle_types[obs_type]} Obstacles:")
        static_q = Q[:, :, obs_type, :]
        
        # Find states with highest and lowest Q-values
        max_q_state = np.unravel_index(np.argmax(static_q), static_q.shape)
        min_q_state = np.unravel_index(np.argmin(static_q), static_q.shape)
        
        print(f"  Highest Q-value: {static_q[max_q_state]:.3f} at state {max_q_state}")
        print(f"  Lowest Q-value: {static_q[min_q_state]:.3f} at state {min_q_state}")
        
        # Analyze action preferences
        best_actions = np.argmax(static_q, axis=2)
        action_counts = np.bincount(best_actions.flatten(), minlength=3)
        print(f"  Action preferences: LEFT={action_counts[0]}, RIGHT={action_counts[1]}, STRAIGHT={action_counts[2]}")
        
        # Find most confident states (highest max - mean difference)
        max_q = np.max(static_q, axis=2)
        mean_q = np.mean(static_q, axis=2)
        confidence = max_q - mean_q
        most_confident = np.unravel_index(np.argmax(confidence), confidence.shape)
        print(f"  Most confident state: {most_confident} (confidence: {confidence[most_confident]:.3f})")

def compare_learning_quality():
    """Compare learning quality across different configurations"""
    
    configs = [
        (4, 0.05, "Slow Learning"),
        (4, 0.1, "Balanced Learning"),
        (4, 0.2, "Fast Learning"),
        (8, 0.05, "High Precision Slow"),
        (8, 0.1, "High Precision Balanced"),
        (8, 0.2, "High Precision Fast")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Q-Learning Quality Comparison', fontsize=16)
    
    for idx, (n_bins, alpha, name) in enumerate(configs):
        q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
        if not os.path.exists(q_table_path):
            continue
            
        Q = np.load(q_table_path)
        ax = axes[idx//3, idx%3]
        
        # Calculate learning quality metrics
        all_q_values = Q.flatten()
        max_q_values = np.max(Q, axis=3)
        mean_q_values = np.mean(Q, axis=3)
        confidence = max_q_values - mean_q_values
        
        # Create quality heatmap
        quality_metric = confidence.mean(axis=2)  # Average confidence across obstacle types
        sns.heatmap(quality_metric, annot=True, fmt='.2f', cmap='viridis', ax=ax)
        ax.set_title(f'{name}\n(n_bins={n_bins}, α={alpha})')
        ax.set_xlabel('Angle bins')
        ax.set_ylabel('Distance bins')
    
    plt.tight_layout()
    plt.savefig('results/learning_quality_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print comparison statistics
    print(f"\n{'='*60}")
    print("LEARNING QUALITY COMPARISON")
    print(f"{'='*60}")
    
    for n_bins, alpha, name in configs:
        q_table_path = f"results/bins{n_bins}_alpha{alpha}/q_table.npy"
        if not os.path.exists(q_table_path):
            continue
            
        Q = np.load(q_table_path)
        all_q_values = Q.flatten()
        max_q_values = np.max(Q, axis=3)
        mean_q_values = np.mean(Q, axis=3)
        confidence = max_q_values - mean_q_values
        
        print(f"\n{name} (n_bins={n_bins}, α={alpha}):")
        print(f"  Mean Q-value: {np.mean(all_q_values):.3f}")
        print(f"  Q-value std: {np.std(all_q_values):.3f}")
        print(f"  Average confidence: {np.mean(confidence):.3f}")
        print(f"  Max confidence: {np.max(confidence):.3f}")
        print(f"  Positive Q-values: {(all_q_values > 0).sum() / len(all_q_values) * 100:.1f}%")

if __name__ == "__main__":
    print("Analyzing Q-Table Evolution and Learning Quality...")
    
    # Analyze individual configurations
    print("\nAnalyzing 4x4 balanced learning:")
    analyze_q_table_evolution(n_bins=4, alpha=0.1)
    
    print("\nAnalyzing 8x8 fast learning:")
    analyze_q_table_evolution(n_bins=8, alpha=0.2)
    
    # Compare all configurations
    print("\nComparing learning quality across configurations:")
    compare_learning_quality() 