import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

N = 8
GROUP = 1  # one color per column

# 8 distinct, reasonably muted colors
GROUP_COLORS = ["#4C72B0", "#DD8452", "#55A467", "#C44E52",
                "#8172B3", "#937860", "#DA8BC3", "#CCB974"]


def draw_grid(ax, x0, title, swizzle):
    ax.text(x0 + N / 2, -1.0, title, ha="center", va="bottom",
            fontsize=14, fontweight="bold")
    for r in range(N):
        for c in range(N):
            out_c = c ^ r if swizzle else c
            color = GROUP_COLORS[c // GROUP]
            ax.add_patch(Rectangle((x0 + out_c, r), 1, 1,
                                   facecolor=color, edgecolor="white",
                                   linewidth=0.4))
            if swizzle:
                ax.text(x0 + out_c + 0.5, r + 0.5, f"{c}",
                        ha="center", va="center", fontsize=11,
                        color="black")


fig, ax = plt.subplots(figsize=(10, 5))
gap = 2
ax.set_xlim(-1, 2 * N + gap + 1)
ax.set_ylim(-2, N + 1)
ax.invert_yaxis()
ax.set_aspect("equal")
ax.axis("off")

draw_grid(ax, 0, "input (row, col)", swizzle=False)
draw_grid(ax, N + gap, "output: col -> col ^ row", swizzle=True)

# arrow between the two grids
ax.annotate("", xy=(N + gap - 0.2, N / 2), xytext=(N + 0.2, N / 2),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.text(N + gap / 2, N / 2 - 0.8, "XOR row", ha="center", fontsize=11)

plt.tight_layout()
plt.savefig(f"diagrams/swizzle_xor_2d_{N}x{N}.png", dpi=200, bbox_inches="tight")
