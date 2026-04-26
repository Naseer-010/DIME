import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.facecolor'] = '#fdfdfd'
plt.rcParams['axes.facecolor'] = '#fdfdfd'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

tasks = [
    "connection_pool_deadlock", "node_failure", "retry_storm", "thundering_herd",
    "traffic_spike", "level_5_alibaba_trace", "split_brain_io", "autoscaler_trap",
    "hot_shard_skew", "zombie_node", "black_swan_az", "mem_leak_burn"
]
# Reversing so the biggest gains are at the top of the horizontal chart
tasks.reverse()

zero_shot = np.array([0.990, 0.438, 0.458, 0.435, 0.376, 0.429, 0.415, 0.024, 0.393, 0.377, 0.220, 0.630])
fine_tuned = np.array([0.990, 0.565, 0.514, 0.493, 0.471, 0.531, 0.534, 0.399, 0.606, 0.587, 0.920, 0.976])

y = np.arange(len(tasks))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 8))

bars1 = ax.barh(y - height/2, zero_shot, height, label='Zero-shot', color='#cccccc', edgecolor='#333333', linewidth=0.5)
bars2 = ax.barh(y + height/2, fine_tuned, height, label='Fine-tuned (GRPO)', color='#4682b4', edgecolor='#333333', linewidth=0.5)

ax.set_xlabel('Task Score', fontsize=12)
ax.set_title('DIME Benchmark: Zero-shot vs. GRPO Fine-tuned', fontsize=14, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(tasks, fontsize=10)
ax.legend(frameon=False, loc='lower right')

# Add small delta text annotations
for i, (zs, ft) in enumerate(zip(zero_shot, fine_tuned)):
    delta = ft - zs
    if delta > 0.01:
        ax.text(ft + 0.02, y[i] + height/2, f"+{delta:.2f}", va='center', fontsize=9, color='#333333')

ax.grid(True, axis='x', linestyle='--', alpha=0.3)
ax.set_xlim(0, 1.15)

plt.tight_layout()
plt.savefig('fig7_benchmark_comparison.png', dpi=300, bbox_inches='tight')
print("Saved fig7_benchmark_comparison.png")