import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import sys
import os
# usage: python figures/cast_samples.py <GDXray+ Castings folder> <patch folder from make_casting_patches.py>
#        (writes cast_samples.jpg)
R = sys.argv[1] if len(sys.argv) > 1 else "Castings"
P = sys.argv[2] if len(sys.argv) > 2 else "casting_patches"
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "castings", "casting_patches.csv"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
fig = plt.figure(figsize=(7.2, 4.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1], hspace=0.12)
top = gs[0].subgridspec(1, 3, wspace=0.05)
for j, (s, lab) in enumerate([("C0001", "wheel (C0001)"), ("C0045", "wheel (C0045)"), ("C0055", "knuckle (C0055)")]):
    ax = fig.add_subplot(top[j]); I = np.asarray(Image.open(f"{R}/{s}/{s}_0001.png").convert("L"))
    ax.imshow(I, cmap="gray", vmin=0, vmax=255)
    gt = np.loadtxt(f"{R}/{s}/ground_truth.txt", ndmin=2)
    for _, x1, x2, y1, y2 in gt[gt[:, 0] == 1]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#ff3b30", lw=1.0))
    ax.set_title(lab, fontsize=7); ax.axis("off")
fig.text(0.095, 0.885, "(a)", fontsize=8, fontweight="bold")
bot = gs[1].subgridspec(2, 8, wspace=0.04, hspace=0.06)
pick = ["C0010", "C0034", "C0035", "C0039", "C0045", "C0051", "C0055", "C0061"]
for j, s in enumerate(pick):
    d = df[(df.series == s) & (df.label == 1)].reset_index(drop=True); k = len(d) // 2
    f_def = d.file[k]; kk = f_def.split("_d")[-1]
    f_cln = df[(df.series == s) & (df.label == 0) & df.file.str.endswith("_c" + kk)].file.iloc[0]
    for r, f in enumerate([f_def, f_cln]):
        ax = fig.add_subplot(bot[r, j]); ax.imshow(Image.open(f"{P}/img/{f}").convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color("#ff3b30" if r == 0 else "#1baf7a"); sp.set_linewidth(1.2)
        if r == 0: ax.set_title(s, fontsize=6.5)
        if j == 0: ax.set_ylabel("defective" if r == 0 else "clean", fontsize=7)
fig.text(0.095, 0.455, "(b)", fontsize=8, fontweight="bold")
fig.savefig("cast_samples.jpg", dpi=220, bbox_inches="tight", pil_kwargs={"quality": 90})
