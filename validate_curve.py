import matplotlib.pyplot as plt
import numpy as np

# Data points
epochs = [1, 2, 3, 4]
validation_scores = [0.25, 0.35, 0.3, 0.3]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the validation curve
plt.plot(epochs, validation_scores, 'b-o', linewidth=2, markersize=8, 
         label='Validation Score', markerfacecolor='blue', markeredgewidth=2)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle='--')

# Labels and title
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Validation Score', fontsize=12, fontweight='bold')
plt.title('Epoch Validation Curve', fontsize=14, fontweight='bold')

# Set axis limits with some padding
plt.xlim(0.5, 4.5)
plt.ylim(0.2, 0.4)

# Add value annotations on each point
# for i, (x, y) in enumerate(zip(epochs, validation_scores)):
#     plt.annotate(f'{y:.2f}', 
#                 xy=(x, y), 
#                 xytext=(0, 10),
#                 textcoords='offset points',
#                 ha='center',
#                 fontsize=10,
#                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

# Add legend
plt.legend(loc='best', fontsize=10)

# Customize tick marks
plt.xticks(epochs)
plt.yticks(np.arange(0.2, 0.41, 0.05))

# Add a horizontal line at y=0.3 to show convergence
plt.axhline(y=0.3, color='r', linestyle=':', alpha=0.5, label='y=0.3')

# Tight layout for better spacing
plt.tight_layout()

# Display the plot
plt.show()

# Optional: Save the figure
plt.savefig('epoch_validation_curve.png', dpi=300, bbox_inches='tight')
print("Plot saved as 'epoch_validation_curve.png'")