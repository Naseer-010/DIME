import numpy as np
import matplotlib.pyplot as plt

# 3. Dynamic Cost Coupling (Efficiency vs. Stability)
nodes_active = np.linspace(10, 100, 100)
n_max = 100.0
w_cost = 5.0
kappa_lat = 0.5
tau_lat = 50.0

# Calculate penalties under different latency conditions
def cost_penalty(nodes, lat):
    return -w_cost * (nodes / n_max) * (1 + kappa_lat * (lat / tau_lat))

cost_normal = cost_penalty(nodes_active, lat=10)   # Healthy system
cost_warning = cost_penalty(nodes_active, lat=50)  # At Threshold
cost_meltdown = cost_penalty(nodes_active, lat=150) # Meltdown

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(nodes_active, cost_normal, label='Healthy State (10ms Latency)', color='green', linewidth=2)
ax.plot(nodes_active, cost_warning, label='SLA Threshold (50ms Latency)', color='orange', linewidth=2, linestyle='-.')
ax.plot(nodes_active, cost_meltdown, label='Meltdown State (150ms Latency)', color='red', linewidth=3, linestyle='--')

ax.set_xlabel('Active Nodes Provisioned', fontsize=12)
ax.set_ylabel('Resource Cost Penalty', fontsize=12)
ax.set_title('Dynamic Cost Coupling: Forcing the Agent to Scale During Crises', fontsize=14, fontweight='bold')

# Annotation
ax.annotate('Cost of nodes becomes exponentially\nworse if latency is high, forcing\nthe agent to scale up.', 
            xy=(80, -10), xytext=(40, -14),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11)

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig3_cost_latency_coupling.png', dpi=300)
print("Saved fig3_cost_latency_coupling.png")
