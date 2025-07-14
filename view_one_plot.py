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
print('Press Enter to see next plot, or type "q" to quit')

for i, plot_path in enumerate(plot_files):
    print(f'\n[{i+1}/{len(plot_files)}] Displaying: {plot_path}')
    
    img = mpimg.imread(plot_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(os.path.basename(plot_path))
    plt.show()
    
    user_input = input('Press Enter for next plot, or "q" to quit: ')
    plt.close()
    
    if user_input.lower() == 'q':
        break

print('Finished viewing plots!') 