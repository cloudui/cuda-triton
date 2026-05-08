import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N = 32
left = [1] * N
right = [1 ^ n for n in range(N)]

fig, ax = plt.subplots(figsize=(4, 10))
ax.set_xlim(-1, 6)
ax.set_ylim(-1, N)
ax.invert_yaxis()
ax.axis("off")


def draw_col(x, values, title):
    ax.text(x + 0.5, -0.6, title, ha="center", va="bottom",
            fontsize=12, fontweight="bold")
    for i, v in enumerate(values):
        ax.add_patch(Rectangle((x, i), 1, 1, fill=False, lw=1.2))
        ax.text(x + 0.5, i + 0.5, str(v), ha="center", va="center",
                fontsize=9, family="monospace")
        if x == 0:
            ax.text(-0.4, i + 0.5, str(i), ha="right", va="center",
                    fontsize=8, color="gray")


draw_col(0, left, "bank")
draw_col(4, right, "bank ^ row")

ax.annotate("", xy=(3.9, N / 2), xytext=(1.1, N / 2),
            arrowprops=dict(arrowstyle="->", lw=1.5))
ax.text(2.5, N / 2 - 0.6, "XOR row", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("diagrams/swizzle_xor.png", dpi=200, bbox_inches="tight")
