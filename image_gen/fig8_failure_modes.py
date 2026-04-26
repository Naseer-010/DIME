import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = '#fdfdfd'
plt.rcParams['axes.facecolor'] = '#fdfdfd'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

steps = np.linspace(0, 300, 300)

# Simulate the data
run3_std = np.ones(60)
run3_steps = np.linspace(0, 60, 60)

run5_std = np.concatenate([np.random.normal(0.05, 0.02, 80), np.linspace(0.05, 1.0, 40)])
run5_steps = np.linspace(0, 119, 120)

run6_std = np.random.normal(0.04, 0.01, 300)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(run3_steps, run3_std, color='#8b0000', linewidth=2, label='Run 3 (Truncation Failure)')
ax.plot(run5_steps, run5_std, color='#b8860b', linewidth=2, label='Run 5 (Exploded Gradients)')
ax.plot(steps, run6_std, color='#4682b4', linewidth=2, label='Run 6 (Converged Successfully)')

ax.axhline(1.0, color='#333333', linestyle=':', label='Collapse Threshold')

ax.set_title('Training Failure Modes (frac_reward_zero_std)', fontsize=13, fontweight='bold')
ax.set_xlabel('Training Step', fontsize=11)
ax.set_ylabel('Zero Standard Deviation Fraction', fontsize=11)
ax.legend(frameon=False)
ax.set_ylim(-0.1, 1.2)
ax.grid(True, axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('fig8_failure_modes.png', dpi=300, bbox_inches='tight')
print("Saved fig8_failure_modes.png")