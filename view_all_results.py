import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Path to results directory
results_dir = 'results'

# Find all PNG files in all subfolders (bins*_alpha*)
pattern = os.path.join(results_dir, 'bins*_alpha*', '*.png')
plot_files = sorted(glob.glob(pattern))

if not plot_files:
    print('No plots found in results folders!')
    exit(1)

print(f'Found {len(plot_files)} plot images.')

for plot_path in plot_files:
    print(f'Displaying: {plot_path}')
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(plot_path))
    plt.show()
    input('Press Enter to continue to the next plot...')
    plt.close()

print('All plots displayed!') 