import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# Path to results directory
results_dir = 'results'

# Find all PNG files in all subfolders (bins*_alpha*)
pattern = os.path.join(results_dir, 'bins*_alpha*', '*.png')
plot_files = sorted(glob.glob(pattern))

if not plot_files:
    print('No plots found in results folders!')
    exit(1)

print(f'Found {len(plot_files)} plot images.')

# Calculate grid dimensions
n_plots = len(plot_files)
cols = 4  # Show 4 plots per row
rows = math.ceil(n_plots / cols)

# Create subplot grid
fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
if rows == 1:
    axes = axes.reshape(1, -1)

# Flatten axes for easier indexing
axes_flat = axes.flatten()

for i, plot_path in enumerate(plot_files):
    if i < len(axes_flat):
        img = mpimg.imread(plot_path)
        axes_flat[i].imshow(img)
        axes_flat[i].axis('off')
        axes_flat[i].set_title(os.path.basename(plot_path), fontsize=8)

# Hide empty subplots
for i in range(n_plots, len(axes_flat)):
    axes_flat[i].axis('off')

plt.tight_layout()
plt.show()

print('All plots displayed in grid layout!') 