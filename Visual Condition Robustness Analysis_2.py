import matplotlib.pyplot as plt
import numpy as np

# Set font to Palatino Linotype
plt.rcParams['font.family'] = 'Palatino Linotype'

# Data from the lighting experiments
lighting_levels = ['-20%', '0°', '+20%', '+40%']
recall = [0.83, 0.85, 0.83, 0.83]
precision = [0.78, 0.80, 0.73, 0.78]
f1_score = [0.80, 0.83, 0.79, 0.80]
accuracy = [0.91, 0.93, 0.91, 0.92]

# Create figure
plt.figure(figsize=(12, 8))

# Use different colors for lighting experiment
colors = ['#9B59B6', '#E67E22', '#1ABC9C', '#34495E']  # Purple, DarkOrange, Turquoise, DarkGray

# Plot lines first
plt.plot(range(len(lighting_levels)), recall, '-', label='Recall', linewidth=2, color=colors[0])
plt.plot(range(len(lighting_levels)), precision, '-', label='Precision', linewidth=2, color=colors[1])
plt.plot(range(len(lighting_levels)), f1_score, '-', label='F1-Score', linewidth=2, color=colors[2])
plt.plot(range(len(lighting_levels)), accuracy, '-', label='Accuracy', linewidth=2, color=colors[3])

# Plot points separately - normal lighting (0%) with filled markers, others with empty markers
for i, lighting in enumerate(lighting_levels):
    if lighting == '0°':  # Normal lighting - filled markers
        plt.plot(i, recall[i], '*', markersize=10, color=colors[0], markerfacecolor=colors[0])
        plt.plot(i, precision[i], 'p', markersize=8, color=colors[1], markerfacecolor=colors[1])
        plt.plot(i, f1_score[i], 'h', markersize=8, color=colors[2], markerfacecolor=colors[2])
        plt.plot(i, accuracy[i], 'v', markersize=8, color=colors[3], markerfacecolor=colors[3])
    else:  # Changed lighting - empty markers
        plt.plot(i, recall[i], '*', markersize=10, color=colors[0], markerfacecolor='white', markeredgewidth=2)
        plt.plot(i, precision[i], 'p', markersize=8, color=colors[1], markerfacecolor='white', markeredgewidth=2)
        plt.plot(i, f1_score[i], 'h', markersize=8, color=colors[2], markerfacecolor='white', markeredgewidth=2)
        plt.plot(i, accuracy[i], 'v', markersize=8, color=colors[3], markerfacecolor='white', markeredgewidth=2)

plt.title('Performance Metrics under Different Lighting Conditions', fontsize=20, fontweight='bold')
plt.xlabel('Lighting Condition', fontsize=16)
plt.ylabel('Metrics Scores', fontsize=16)
plt.legend(fontsize=14)
plt.grid(True, alpha=0.3)

# Set x-axis to show lighting conditions
plt.xticks(range(len(lighting_levels)), lighting_levels, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(-0.5, len(lighting_levels)-0.5)
plt.ylim(0.70, 0.95)

# Add value annotations
for i, lighting in enumerate(lighting_levels):
    plt.annotate(f'{recall[i]:.3f}', (i, recall[i]),
                textcoords="offset points", xytext=(0,12), ha='center', fontsize=14, fontweight='bold')
    plt.annotate(f'{precision[i]:.3f}', (i, precision[i]),
                textcoords="offset points", xytext=(0,-18), ha='center', fontsize=14, fontweight='bold')
    plt.annotate(f'{f1_score[i]:.3f}', (i, f1_score[i]),
                textcoords="offset points", xytext=(8,8), ha='center', fontsize=14, fontweight='bold')
    plt.annotate(f'{accuracy[i]:.3f}', (i, accuracy[i]),
                textcoords="offset points", xytext=(-8,8), ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Print summary statistics
print("Performance Summary:")
normal_idx = lighting_levels.index('0°')  # Find normal lighting index
worst_recall = min(recall)
worst_precision = min(precision)
worst_f1 = min(f1_score)
worst_accuracy = min(accuracy)

print(f"Recall: {recall[normal_idx]:.3f} → worst: {worst_recall:.3f} (drop: {((recall[normal_idx]-worst_recall)/recall[normal_idx]*100):.1f}%)")
print(f"Precision: {precision[normal_idx]:.3f} → worst: {worst_precision:.3f} (drop: {((precision[normal_idx]-worst_precision)/precision[normal_idx]*100):.1f}%)")
print(f"F1-Score: {f1_score[normal_idx]:.3f} → worst: {worst_f1:.3f} (drop: {((f1_score[normal_idx]-worst_f1)/f1_score[normal_idx]*100):.1f}%)")
print(f"Accuracy: {accuracy[normal_idx]:.3f} → worst: {worst_accuracy:.3f} (drop: {((accuracy[normal_idx]-worst_accuracy)/accuracy[normal_idx]*100):.1f}%)")