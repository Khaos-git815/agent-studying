import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Path to results directory
results_dir = 'results'

# Find all configuration folders
config_folders = sorted(glob.glob(os.path.join(results_dir, 'bins*_alpha*')))

if not config_folders:
    print('No configuration folders found!')
    exit(1)

print(f'Found {len(config_folders)} configurations.')

# Calculate grid dimensions
n_configs = len(config_folders)
cols = 3  # 3 plots per configuration: heatmap, 3D surface, reward
rows = n_configs

# Create subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(18, 4*rows))

# For each configuration, show its 3 main plots
for i, config_folder in enumerate(config_folders):
    config_name = os.path.basename(config_folder)
    print(f'Processing: {config_name}')
    
    # Find the 3 main plot types for this configuration
    heatmap_static = os.path.join(config_folder, 'q_table_plot.png')
    surface_static = os.path.join(config_folder, 'q_table_surface_static.png')
    reward_plot = os.path.join(config_folder, 'reward_plot.png')
    
    # Plot 1: Q-table heatmap (static)
    if os.path.exists(heatmap_static):
        img = mpimg.imread(heatmap_static)
        axes[i, 0].imshow(img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f'{config_name}\nQ-table Heatmap (Static)', fontsize=10)
    
    # Plot 2: 3D surface plot (static)
    if os.path.exists(surface_static):
        img = mpimg.imread(surface_static)
        axes[i, 1].imshow(img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f'{config_name}\n3D Surface (Static)', fontsize=10)
    
    # Plot 3: Reward plot
    if os.path.exists(reward_plot):
        img = mpimg.imread(reward_plot)
        axes[i, 2].imshow(img)
        axes[i, 2].axis('off')
        axes[i, 2].set_title(f'{config_name}\nReward Progress', fontsize=10)

plt.tight_layout()
plt.show()

print('All configurations displayed in grid layout!')