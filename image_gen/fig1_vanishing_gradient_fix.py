import numpy as np
import matplotlib.pyplot as plt

# 1. The Vanishing Gradient Fix (Latency)
lat = np.linspace(0, 150, 500)
tau = 50.0
alpha = 0.01
beta = 5.0
gamma = 0.1

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

# Bad: Flat Sigmoid (Gradient vanishes after 100ms)
bad_reward = -beta * sigmoid(gamma * (lat - tau))

# Good: Linear-Coupled Sigmoid (Gradient survives)
good_reward = -alpha * lat - beta * lat * sigmoid(gamma * (lat - tau))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lat, bad_reward, label='Naive Sigmoid (Gradient Vanishes)', linewidth=3, linestyle='--', color='red')
ax.plot(lat, good_reward, label='Linear-Coupled Sigmoid (DIME Final)', linewidth=3, color='blue')

ax.axvline(tau, color='black', linestyle=':', alpha=0.7, label='SLA Threshold (50ms)')
ax.axvspan(100, 150, color='red', alpha=0.1, label='Panic Zone (Dead Gradients for Naive)')

ax.set_xlabel('System Latency (ms)', fontsize=12)
ax.set_ylabel('Reward Penalty', fontsize=12)
ax.set_title('Solving the Vanishing Gradient in Queue Saturation Penalties', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_vanishing_gradient_fix.png', dpi=300)
print("Saved fig1_vanishing_gradient_fix.png")
