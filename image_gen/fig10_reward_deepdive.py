import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig = plt.figure(figsize=(14, 6))

# --- LEFT PANEL: Line Chart ---
ax1 = fig.add_subplot(121)
steps = np.arange(1, 51)

# Mock data based on the prompt description
r_total = np.interp(steps, [1, 2, 15, 50], [3, 1, 6, 10]) + np.random.normal(0, 0.2, 50)
r_env = np.interp(steps, [1, 2, 15, 50], [-3.5, -3.5, 3, 5]) + np.random.normal(0, 0.2, 50)
r_fmt = np.full(50, 3.0)
r_val = np.full(50, 1.9)
r_tri = np.random.uniform(0.5, 1.0, 50)

ax1.plot(steps, r_total, color='#008000', lw=3, label='Total Reward (R)')
ax1.plot(steps, r_env, color='#008080', lw=2, label='Environment (R_env)')
ax1.plot(steps, r_fmt, color='#0000cc', lw=2, linestyle='--', label='Format (R_fmt)')
ax1.plot(steps, r_val, color='#800080', lw=2, linestyle='--', label='Validity (R_val)')
ax1.plot(steps, r_tri, color='#cc6600', lw=2, linestyle=':', label='Triage (R_tri)')

ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(steps, r_env, 0, where=(r_env>0), color='#008080', alpha=0.1)
ax1.fill_between(steps, r_env, 0, where=(r_env<0), color='#cc0000', alpha=0.1)

ax1.annotate('throttle(0.3) at step 2\n→ env recovers', xy=(4, -3), xytext=(10, -4.5),
             arrowprops=dict(arrowstyle="->", color='black'), fontsize=10)

ax1.set_title('Reward Breakdown: Sample Episode', fontsize=13, fontweight='bold')
ax1.set_xlabel('Episode Step', fontsize=11)
ax1.set_ylabel('Reward Value', fontsize=11)
ax1.legend(loc='upper left', fontsize=9)

# --- RIGHT PANEL: Radar Chart ---
ax2 = fig.add_subplot(122, polar=True)
categories = ['Format', 'Action Val.', 'DB Recov.', 'Load Shed', 'Latency', 'Triage']
N = len(categories)

zero_shot = [0.90, 0.75, 0.28, 0.55, 0.48, 0.35]
fine_tuned = [0.99, 0.97, 0.88, 0.82, 0.79, 0.76]

# Close the loops
zero_shot += zero_shot[:1]
fine_tuned += fine_tuned[:1]
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax2.plot(angles, zero_shot, color='#888888', linestyle='--', linewidth=2, label='Zero-shot')
ax2.fill(angles, zero_shot, color='#888888', alpha=0.1)

ax2.plot(angles, fine_tuned, color='#008000', linewidth=2, label='Fine-tuned (GRPO)')
ax2.fill(angles, fine_tuned, color='#008000', alpha=0.25)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=11)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color="gray", size=8)
ax2.set_title('Capability Radar Comparison', y=1.08, fontsize=13, fontweight='bold')
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig('fig10_reward_deepdive.png', dpi=300)
print("Saved fig10_reward_deepdive.png")