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
print('Showing 2 plots at a time.')
print('Press Enter to see next pair, or type "q" to quit')

# Calculate number of pairs
n_pairs = math.ceil(len(plot_files) / 2)

for pair_idx in range(n_pairs):
    start_idx = pair_idx * 2
    end_idx = min(start_idx + 2, len(plot_files))
    
    print(f'\n[Pair {pair_idx+1}/{n_pairs}] Displaying plots {start_idx+1}-{end_idx}')
    
    # Create subplot with 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Show first plot
    if start_idx < len(plot_files):
        img1 = mpimg.imread(plot_files[start_idx])
        axes[0].imshow(img1)
        axes[0].axis('off')
        axes[0].set_title(os.path.basename(plot_files[start_idx]), fontsize=10)
    
    # Show second plot (if exists)
    if start_idx + 1 < len(plot_files):
        img2 = mpimg.imread(plot_files[start_idx + 1])
        axes[1].imshow(img2)
        axes[1].axis('off')
        axes[1].set_title(os.path.basename(plot_files[start_idx + 1]), fontsize=10)
    else:
        # Hide second subplot if no second plot
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    user_input = input('Press Enter for next pair, or "q" to quit: ')
    plt.close()
    
    if user_input.lower() == 'q':
        break

print('Finished viewing plots!')
