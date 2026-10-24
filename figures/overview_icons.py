# Fig. "Overview of the benchmark" (icon version). usage: python figures/overview_icons.py  (writes overview_icons.pdf)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Ellipse, FancyArrowPatch, Polygon

plt.rcParams.update({"font.family": "DejaVu Sans"})
INK, MUTE, GRID = "#2b2b2b", "#6b6a66", "#c9c8c2"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#d64532"
W, H = 72, 27
fig = plt.figure(figsize=(7.2, 2.7))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
XS = [5.2, 15.4, 25.6, 35.8, 46.0, 56.2, 66.4]
YC = 13.0  # icon centre
rng = np.random.default_rng(3)


def tile(x, title, sub, fc="#f5f4f0", ec="#d6d5cf"):
    ax.add_patch(FancyBboxPatch((x - 4.5, 2.2), 9.0, 17.0, boxstyle="round,pad=0,rounding_size=1.0", fc=fc, ec=ec, lw=0.8))
    ax.text(x, 6.0, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color=INK)
    ax.text(x, 3.6, sub, ha="center", va="center", fontsize=6.3, color=MUTE)


def arrow(x0, x1, y=YC, **kw):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=kw.get("c", INK),
                                 ls=kw.get("ls", "-"), connectionstyle=kw.get("cs", "arc3")))


# 1 radiographs: weld film + casting wheel
x = XS[0]; tile(x, "Radiographs", "welds · castings")
ax.add_patch(FancyBboxPatch((x - 3.6, YC - 0.4), 5.0, 3.6, boxstyle="round,pad=0,rounding_size=0.4", fc="#3b3b3b", ec="none"))
ax.add_patch(Rectangle((x - 3.6, YC + 0.8), 5.0, 1.2, fc="#9a9a9a", ec="none"))
ax.add_patch(Circle((x - 1.4, YC + 1.4), 0.32, fc="white", ec="none"))
cx, cy = x + 1.6, YC - 1.6
ax.add_patch(Circle((cx, cy), 2.3, fc="#d9d8d2", ec="#5c5c58", lw=1.0))
ax.add_patch(Circle((cx, cy), 0.6, fc="#5c5c58", ec="none"))
for a in np.linspace(0, 2 * np.pi, 6, endpoint=False):
    ax.plot([cx + 0.6 * np.cos(a), cx + 2.2 * np.cos(a)], [cy + 0.6 * np.sin(a), cy + 2.2 * np.sin(a)], color="#5c5c58", lw=1.1)
ax.add_patch(Circle((cx + 1.2, cy + 1.2), 0.3, fc=RED, ec="none"))

# 2 labelled samples: grid of patches, defective outlined red, clean green
x = XS[1]; tile(x, "Samples", "sections · patches")
for i in range(3):
    for j in range(3):
        g = rng.uniform(0.35, 0.8)
        bad = (i + j) % 2 == 0
        ax.add_patch(Rectangle((x - 3.3 + 2.3 * j, YC - 2.6 + 2.3 * i), 2.0, 2.0, fc=(g, g, g), ec=RED if bad else GREEN, lw=1.1))
        if bad:
            ax.add_patch(Circle((x - 3.3 + 2.3 * j + 1.0, YC - 2.6 + 2.3 * i + 1.0), 0.35, fc="white", ec="none", alpha=0.9))

# 3 features: CNN layer stack
x = XS[2]; tile(x, "Features", "HOG · ResNet-18")
for k, (hh, ww) in enumerate([(6.0, 1.0), (4.8, 1.0), (3.6, 1.0), (2.4, 1.0)]):
    xx = x - 3.2 + k * 1.75
    ax.add_patch(Polygon([[xx, YC - hh / 2], [xx + ww, YC - hh / 2 + 0.5], [xx + ww, YC + hh / 2 + 0.5], [xx, YC + hh / 2]],
                         closed=True, fc=["#c7dcf5", "#a3c6ef", "#7eafe8", "#5a98e0"][k], ec="#3d6fae", lw=0.7))
ax.add_patch(Circle((x + 3.4, YC + 0.3), 0.35, fc="#3d6fae", ec="none"))

# 4 PCA: point cloud with principal axes
x = XS[3]; tile(x, "PCA", "d = 4–12 qubits")
pts = rng.multivariate_normal([0, 0], [[3.0, 1.9], [1.9, 1.6]], 60)
pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < 3.4][:40]
ax.scatter(x + pts[:, 0] * 0.95, YC + pts[:, 1] * 0.95, s=3, color="#8a8984", zorder=3)
v = np.array([0.83, 0.56])
ax.add_patch(FancyArrowPatch((x, YC), (x + 3.6 * v[0], YC + 3.6 * v[1]), arrowstyle="-|>", mutation_scale=7, lw=1.2, color=INK, zorder=4))
ax.add_patch(FancyArrowPatch((x, YC), (x - 1.8 * v[1], YC + 1.8 * v[0]), arrowstyle="-|>", mutation_scale=7, lw=1.2, color=INK, zorder=4))

# 5 kernel: classical matrix (top) + quantum circuit (bottom)
x = XS[4]; tile(x, "Kernel", "classical · quantum", fc="#eaf2fc", ec=BLUE)
M = np.exp(-((np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2) / 3.0)
for i in range(5):
    for j in range(5):
        ax.add_patch(Rectangle((x - 3.6 + 0.75 * j, YC + 0.4 + 0.75 * (4 - i)), 0.75, 0.75,
                               fc=plt.cm.Blues(0.15 + 0.8 * M[i, j]), ec="white", lw=0.3))
for q in range(3):
    yq = YC - 1.0 - 1.1 * q
    ax.plot([x - 0.6, x + 3.8], [yq, yq], color=INK, lw=0.8)
    ax.add_patch(Rectangle((x - 0.3, yq - 0.35), 0.7, 0.7, fc="white", ec=INK, lw=0.7))
ax.plot([x + 1.6, x + 1.6], [YC - 1.0, YC - 2.1], color=INK, lw=0.8)
ax.add_patch(Circle((x + 1.6, YC - 1.0), 0.2, fc=INK, ec="none"))
ax.add_patch(Circle((x + 1.6, YC - 2.1), 0.32, fc="white", ec=INK, lw=0.8))
ax.plot([x + 1.6, x + 1.6], [YC - 2.42, YC - 1.78], color=INK, lw=0.7)
ax.add_patch(Rectangle((x + 2.4, YC - 3.55), 0.9, 0.9, fc="white", ec=INK, lw=0.7))
ax.text(x - 3.6, YC + 4.6, "RBF", fontsize=5.2, color=MUTE, va="bottom")
ax.text(x + 3.9, YC - 4.6, "ZZ", fontsize=5.2, color=MUTE, ha="right", va="top")

# 6 classifier: two classes + curved boundary
x = XS[5]; tile(x, "SVM", "validation-tuned")
a = rng.normal([-1.6, 1.2], 0.75, (12, 2)); b = rng.normal([1.5, -1.0], 0.75, (12, 2))
ax.scatter(x + a[:, 0], YC + a[:, 1], s=7, color=GREEN, zorder=3)
ax.scatter(x + b[:, 0], YC + b[:, 1], s=7, color=RED, marker="s", zorder=3)
t = np.linspace(-3.6, 3.6, 50)
ax.plot(x + t, YC + 0.15 * t ** 2 * np.sign(t) * 0.5 + 0.55 * t - 0.2, color=INK, lw=1.1)

# 7 evaluation: ROC curve
x = XS[6]; tile(x, "Evaluation", "grouped CV · AUC")
x0, y0, s = x - 3.0, YC - 3.2, 6.2
ax.plot([x0, x0, x0 + s], [y0 + s, y0, y0], color=INK, lw=0.9)
ax.plot([x0, x0 + s], [y0, y0 + s], color=GRID, lw=0.8, ls=(0, (2, 2)))
u = np.linspace(0, 1, 50)
ax.plot(x0 + s * u, y0 + s * (1 - (1 - u) ** 3.2), color=BLUE, lw=1.4)
ax.fill_between(x0 + s * u, y0, y0 + s * (1 - (1 - u) ** 3.2), color=BLUE, alpha=0.12, lw=0)

for i in range(6):
    arrow(XS[i] + 4.6, XS[i + 1] - 4.6)

# cut-paste (welds only): small icon above, dashed into features
xc, yc = (XS[1] + XS[2]) / 2, 23.6
ax.add_patch(Rectangle((xc - 1.0, yc - 1.0), 2.0, 2.0, fc="#b9b9b5", ec=ORANGE, lw=1.0, ls="--"))
ax.add_patch(Circle((xc + 0.2, yc + 0.1), 0.38, fc="white", ec="none"))
ax.text(xc + 1.6, yc, "cut-paste (welds)", fontsize=6.2, color=ORANGE, va="center", style="italic")
ax.add_patch(FancyArrowPatch((xc - 1.0, yc), (XS[2] - 3.3, 19.3), arrowstyle="-|>", mutation_scale=7, lw=0.9,
                             color=ORANGE, ls=(0, (3, 2)), connectionstyle="arc3,rad=0.25"))

fig.savefig("overview_icons.pdf", bbox_inches="tight", pad_inches=0.02)
