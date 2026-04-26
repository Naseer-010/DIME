import numpy as np
import matplotlib.pyplot as plt

# Minimalistic, clean aesthetic settings
plt.rcParams['figure.facecolor'] = '#fdfdfd'
plt.rcParams['axes.facecolor'] = '#fdfdfd'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['text.color'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#333333'
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

x = np.linspace(0, 1.0, 500)

# Left Panel: Original Cliff
y_original = -100 * np.exp(4 * x)
cliff_mask = x >= 0.57
y_original[cliff_mask] = -1000

# Right Panel: Redesigned Bounded
y_redesigned = np.clip(-1.5 * np.exp(2 * x) + 1.0 - 0.05, -5.0, 5.0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Reward Function Redesign: Eliminating the Gradient-Blocking Cliff', fontsize=14, fontweight='bold', y=1.05)

# Plot 1: Cliff
ax1.plot(x, y_original, color='#8b0000', linewidth=2)
ax1.axvline(0.57, color='#333333', linestyle=':', label='Cliff Threshold (0.57)')
ax1.fill_between(x, y_original, -1050, where=cliff_mask, color='#8b0000', alpha=0.1)
ax1.set_title('Original Reward (Gradient-Blocking)', fontsize=12)
ax1.set_xlabel('DB Node CPU Utilization', fontsize=11)
ax1.set_ylabel('Reward', fontsize=11)
ax1.set_ylim(-1100, 50)
ax1.legend(frameon=False)

# Plot 2: Bounded
ax2.plot(x, y_redesigned, color='#4682b4', linewidth=2)
ax2.axhline(-5.0, color='#333333', linestyle=':', label='Clip Bounds (±5)')
ax2.axhline(5.0, color='#333333', linestyle=':')
ax2.fill_between(x, y_redesigned, 0, where=(y_redesigned < 0), color='#4682b4', alpha=0.1)
ax2.fill_between(x, y_redesigned, 0, where=(y_redesigned > 0), color='#556b2f', alpha=0.1)
ax2.set_title('Redesigned Reward (Bounded & Differentiable)', fontsize=12)
ax2.set_xlabel('DB Node CPU Utilization', fontsize=11)
ax2.set_ylim(-6, 6)
ax2.legend(frameon=False)

plt.tight_layout()
plt.savefig('fig5_reward_cliff_bounded.png', dpi=300, bbox_inches='tight')
print("Saved fig5_reward_cliff_bounded.png")