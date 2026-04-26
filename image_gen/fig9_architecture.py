import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Clean light theme
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off') # Hide grid and axes

# --- Draw Components ---
# 1. LLM Agent (Top)
ax.add_patch(patches.Rectangle((2.5, 8), 5, 1.5, edgecolor='#333333', facecolor='#e6f2ff', lw=2, zorder=2))
ax.text(5, 8.9, "LLM Agent (Qwen3-8B + LoRA)", ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(5, 8.4, "<reasoning>...</reasoning>\n<action>kubectl...</action>", ha='center', va='center', fontsize=10, color='#555555')

# 2. Ingress / Load Balancer (Middle)
ax.add_patch(patches.Rectangle((3.5, 5.5), 3, 1, edgecolor='#333333', facecolor='#fff3e6', lw=2, zorder=2))
ax.text(5, 6.1, "DIME", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 5.7, "throttle(rate) | error_budget", ha='center', va='center', fontsize=9, color='#555555')

# 3. Database Node-0 (Bottom Left)
ax.add_patch(patches.Rectangle((0.5, 1), 2.5, 2.5, edgecolor='#cc0000', facecolor='#ffeeee', lw=2.5, zorder=2))
ax.text(1.75, 2.5, "node-0", ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.75, 1.8, "PostgreSQL DB\n(SPOF)", ha='center', va='center', fontsize=10)
ax.text(1.75, 1.3, "⚠ Single Point of Failure", ha='center', va='center', fontsize=9, color='#cc0000', fontweight='bold')

# 4. Worker Nodes 1-7 (Bottom Right)
for i in range(1, 8):
    x_pos = 3.5 + (i-1)*0.9
    color = '#e6ffe6' if i <= 5 else '#ffeeee'
    edge = '#008000' if i <= 5 else '#cc0000'
    label = f"n-{i}"
    status = "worker" if i <= 5 else "failed"
    
    ax.add_patch(patches.Rectangle((x_pos, 1.5), 0.7, 1.5, edgecolor=edge, facecolor=color, lw=1.5, zorder=2))
    ax.text(x_pos+0.35, 2.5, label, ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(x_pos+0.35, 1.8, status, ha='center', va='center', fontsize=8)

# --- Draw Arrows ---
# Action arrow (Agent -> Ingress)
ax.annotate("", xy=(4.5, 6.5), xytext=(4.5, 8), arrowprops=dict(arrowstyle="->", color='#800080', lw=2))
ax.text(4.4, 7.25, "action $a_t$", ha='right', va='center', fontsize=11, color='#800080', fontweight='bold')

# Observation arrow (Nodes -> Agent)
ax.annotate("", xy=(5.5, 8), xytext=(5.5, 6.5), arrowprops=dict(arrowstyle="->", color='#008080', lw=2))
ax.text(5.6, 7.25, "observation $s_t$", ha='left', va='center', fontsize=11, color='#008080', fontweight='bold')

# Reward arrow (Environment -> Agent)
ax.annotate("", xy=(7.5, 8.75), xytext=(9, 8.75), arrowprops=dict(arrowstyle="->", color='#cc6600', lw=2))
ax.annotate("", xy=(9, 8.75), xytext=(9, 4), arrowprops=dict(arrowstyle="-", color='#cc6600', lw=2))
ax.annotate("", xy=(9, 4), xytext=(7.5, 4), arrowprops=dict(arrowstyle="-", color='#cc6600', lw=2))
ax.text(9.1, 6.5, "Reward $R(s_t, a_t)$", ha='left', va='center', fontsize=11, color='#cc6600', fontweight='bold', rotation=270)

# Connect Ingress to nodes
for i in range(8):
    x_target = 1.75 if i == 0 else 3.85 + (i-1)*0.9
    ax.plot([5, x_target], [5.5, 3.0], color='#aaaaaa', lw=1, zorder=1)

plt.title("DIME — Distributed Infrastructure Management Environment", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('fig9_architecture.png', dpi=300)
print("Saved fig9_architecture.png")