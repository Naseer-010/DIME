import numpy as np
import matplotlib.pyplot as plt

# 4. Intrinsic Reward (RND) Annealing over Training Steps
training_steps = np.linspace(0, 10000, 500)

# Simulate standard extrinsic reward slowly climbing as agent learns
extrinsic_reward = -5.0 + 10.0 * (1 - np.exp(-training_steps / 3000))

# Simulate intrinsic reward (Curiosity) decaying over time
initial_beta = 5.0
intrinsic_multiplier = initial_beta * np.exp(-training_steps / 2000)
# Add some noise to simulate prediction error spikes when finding new states
intrinsic_noise = np.abs(np.random.normal(0, 0.5, len(training_steps))) * intrinsic_multiplier
intrinsic_reward = intrinsic_multiplier + intrinsic_noise

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(training_steps, extrinsic_reward, label='Extrinsic SRE Reward (Task Mastery)', color='blue', linewidth=2)
ax.plot(training_steps, intrinsic_reward, label='Intrinsic RND Bonus (Curiosity)', color='orange', linewidth=2, alpha=0.8)

ax.fill_between(training_steps, intrinsic_reward, alpha=0.2, color='orange')

ax.set_xlabel('GRPO Training Steps', fontsize=12)
ax.set_ylabel('Reward Magnitude', fontsize=12)
ax.set_title('RND Curiosity Annealing: Transitioning from Exploration to Exploitation', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotations
ax.annotate('Agent explores edge-case\ncascading failures here', xy=(1000, 4), xytext=(2000, 6),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11, ha='center')

ax.annotate('Agent exploits the optimal\nSRE policy here', xy=(8000, 4), xytext=(8000, 1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
            fontsize=11, ha='center')

plt.tight_layout()
plt.savefig('fig4_curiosity_annealing.png', dpi=300)
print("Saved fig4_curiosity_annealing.png")
