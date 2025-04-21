import matplotlib.pyplot as plt
import numpy as np

# Data extracted from report/gan_training_summary.txt
samples_per_class = np.array([0, 100, 200, 500])
map_scores = np.array([0.6132, 0.4993, 0.3697, 0.1973])

# Apply a professional style
plt.style.use('seaborn-v0_8-darkgrid') # Or try 'ggplot', 'seaborn-v0_8-whitegrid', etc.

# Create the plot
plt.figure(figsize=(10, 6)) # Slightly larger figure size
plt.plot(samples_per_class, map_scores, marker='o', linestyle='-', linewidth=2, markersize=8)

# Add titles and labels with adjusted font sizes
plt.xlabel('Number of GAN-generated Samples per Class', fontsize=12)
plt.ylabel('Mean Average Precision (mAP)', fontsize=12)

# Improve grid and ticks
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth='0.5', color='lightgrey')
plt.tick_params(axis='both', which='major', labelsize=10)

# Set x-axis ticks to match the data points
plt.xticks(samples_per_class)
# Optional: Set y-axis limits for better focus if needed
# plt.ylim(0, 0.7)

# Add annotations for the data points
for i, txt in enumerate(map_scores):
    plt.annotate(f'{txt:.4f}', 
                 (samples_per_class[i], map_scores[i]), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center', 
                 fontsize=10)

# Add a subtle background color
ax = plt.gca()
ax.set_facecolor('#f7f7f7') # Light gray background

# Ensure tight layout
plt.tight_layout()

# Display the plot
plt.savefig('map_vs_gan_samples.png', dpi=300)

# Optional: Save the plot
# plt.savefig('map_vs_gan_samples.png', dpi=300)
# plt.savefig('map_vs_gan_samples.pdf')
