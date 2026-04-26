import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

fig, ax = plt.subplots(figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Data for the 10 rules
rules = [
    ("OOM: $\exists i: m_i > 0.92$", "restart_node(i)", "#cc6600"),          # 1
    ("DB FAIL: $0 \in F(s)$", "restart_node(0)", "#cc0000"),                # 2
    ("Split-brain: $w_{io} > 0.80$", "throttle(0.5)", "#008080"),           # 3
    ("Hot-shard: $c_i > 0.90, \\bar{c} < 0.60$", "reroute(i $\\to$ j)", "#0000cc"), # 4
    ("Retry storm: $\lambda_{99}>100, r>150$", "throttle(0.4)", "#800080"), # 5
    ("Zombie node: $c_i \in [0, 0.10)$", "reroute(i $\\to$ j)", "#0000cc"), # 6
    ("Black swan: $|F(s)| \geq 2$", "throttle(0.3)", "#cc6600"),            # 7
    ("DB stress: $c_0 > 0.80$", "throttle(0.7)", "#cc6600"),                # 8
    ("Safe scale: $\\bar{c}_{work}>0.75, B>20$", "scale_up", "#008000"),    # 9
    ("Healthy system", "no_op", "#888888")                                  # 10
]

# Title and Entry Box
ax.text(5, 11.5, "Triage Oracle $a^*(s)$ — 10-Rule Priority Decision Tree", ha='center', fontsize=14, fontweight='bold')
ax.add_patch(patches.Rectangle((3, 10.5), 4, 0.6, edgecolor='black', facecolor='#e6f2ff', lw=1.5))
ax.text(5, 10.8, "Observe state $s_t$", ha='center', va='center', fontsize=11, fontweight='bold')
ax.annotate("", xy=(5, 10.5), xytext=(5, 10.2), arrowprops=dict(arrowstyle="->", color='black'))

# Draw the sequence
y_start = 9.8
y_step = 0.9

# Draw Priority Inversion Zone Box (covers rules 2 through 7)
ax.add_patch(patches.Rectangle((0.5, y_start - 6.5*y_step - 0.2), 9, 6.2, edgecolor='#cc0000', facecolor='none', linestyle='--', lw=2))
ax.text(9.4, y_start - 3.5*y_step, "Priority Inversion Zone\n(Fixed in v2)", ha='right', va='center', color='#cc0000', fontsize=10, fontweight='bold')

for i, (cond, act, color) in enumerate(rules):
    y = y_start - i * y_step
    
    # Condition Box (Left)
    ax.add_patch(patches.Rectangle((2, y-0.25), 3.5, 0.5, edgecolor=color, facecolor='white', lw=2))
    ax.text(3.75, y, f"Rule {i+1}: {cond}", ha='center', va='center', fontsize=10)
    
    # Action Box (Right)
    ax.add_patch(patches.Rectangle((6.5, y-0.25), 2.5, 0.5, edgecolor=color, facecolor=color, alpha=0.1, lw=2))
    ax.text(7.75, y, act, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    
    # Arrow from Condition to Action (True)
    ax.annotate("", xy=(6.5, y), xytext=(5.5, y), arrowprops=dict(arrowstyle="->", color=color, lw=2))
    ax.text(6.0, y+0.1, "True", ha='center', va='bottom', fontsize=8, color=color)
    
    # Arrow to next rule (False)
    if i < len(rules) - 1:
        ax.annotate("", xy=(3.75, y-0.25), xytext=(3.75, y-y_step+0.25), 
                    arrowprops=dict(arrowstyle="<-", color='#888888', lw=1.5))
        ax.text(3.85, y-0.45, "False", ha='left', va='center', fontsize=8, color='#888888')

plt.tight_layout()
plt.savefig('fig11_triage_tree.png', dpi=300)
print("Saved fig11_triage_tree.png")