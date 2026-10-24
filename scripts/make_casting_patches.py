"""Build the GDXray+ casting patch dataset used in the QML study.

Input : the GDXray+ Castings folder (C0001 ... C0087) with ground_truth.txt files
        (one row per bounding box: image index, x1, x2, y1, y2; x = column).
Output: casting_patches/ (PNG crops at native resolution) and casting_patches.csv.

Design
- Only series with bounding-box ground truth are used. C0002 (crops of C0001) and
  C0029 (boxes cover the whole frame; acquisition note says "without defect") are excluded.
- Each series is the same part imaged at different angles, so series that may show the
  same physical part are merged into one GROUP; all splits are made by group.
- Defective patch: square crop centred on a box, side = clip(2 x box side, 48, 224) px.
  At most CAP boxes per series, spread evenly over the views.
- Clean patch: for every defective patch, one window of the same size from an image of the
  same series that contains annotated defects, not overlapping any box (8 px margin), on
  material (not saturated background), and closest in mean/std intensity to the defective patch.
  Both classes therefore come from the same parts, views and exposure.
"""
import os, sys, glob, numpy as np, pandas as pd
from PIL import Image

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Castings"
OUT = sys.argv[2] if len(sys.argv) > 2 else "casting_patches"
CAP, SEED, MARGIN, MIN_SIDE = 20, 7, 8, 8
EXCLUDE = {"C0002", "C0029"}

# physical part per series (wheel identifier from the acquisition notes in each readme)
GROUP = {"C0001": "wheel_BMW", "C0007": "wheel_14", "C0043": "wheel_14", "C0008": "wheel_IJ",
         "C0010": "wheel_1128", "C0014": "wheel_L7", "C0015": "wheel_L7", "C0019": "wheel_5",
         "C0021": "wheel_N1", "C0024": "wheel_bed", "C0026": "wheel_TTC", "C0030": "wheel_55",
         "C0031": "wheel_18", "C0032": "wheel_17", "C0033": "wheel_17",
         "C0034": "wheel_q3", "C0035": "wheel_q3", "C0036": "wheel_q3",
         "C0038": "wheel_q4", "C0039": "wheel_q4", "C0040": "wheel_q4",
         "C0041": "wheel_q5", "C0042": "wheel_q5",
         "C0045": "wheel_ts", "C0047": "wheel_ts", "C0051": "wheel_ts",
         "C0054": "knuckle", "C0055": "knuckle", "C0057": "knuckle", "C0060": "knuckle",
         "C0061": "knuckle", "C0062": "knuckle", "C0065": "knuckle", "C0087": "part_C0087"}


def load(path):
    return np.asarray(Image.open(path).convert("L"))


def material(p):
    return float(((p >= 12) & (p <= 243)).mean())


def overlaps(x0, y0, s, boxes, m):
    for bx1, bx2, by1, by2 in boxes:
        if x0 < bx2 + m and x0 + s > bx1 - m and y0 < by2 + m and y0 + s > by1 - m:
            return True
    return False


def iou(a, b, s):
    ix = max(0, min(a[0], b[0]) + s - max(a[0], b[0])); iy = max(0, min(a[1], b[1]) + s - max(a[1], b[1]))
    return ix * iy / (2 * s * s - ix * iy)


def main():
    os.makedirs(os.path.join(OUT, "img"), exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, log = [], []
    for sdir in sorted(glob.glob(os.path.join(ROOT, "C*"))):
        s = os.path.basename(sdir)
        gt = os.path.join(sdir, "ground_truth.txt")
        if s in EXCLUDE or not os.path.isfile(gt):
            continue
        files = sorted(glob.glob(os.path.join(sdir, f"{s}_*.png")))
        a = np.loadtxt(gt, ndmin=2)
        H, W = load(files[0]).shape
        boxes = {}
        for i, x1, x2, y1, y2 in a:
            x1, x2 = max(0, x1), min(W - 1, x2); y1, y2 = max(0, y1), min(H - 1, y2)
            if max(x2 - x1, y2 - y1) < MIN_SIDE or int(i) > len(files):
                continue
            boxes.setdefault(int(i), []).append((x1, x2, y1, y2))
        flat = sorted((i, b) for i, bs in boxes.items() for b in bs)
        pick = np.unique(np.round(np.linspace(0, len(flat) - 1, min(CAP, len(flat)))).astype(int))
        imgs = {i: load(files[i - 1]) for i in boxes}
        used = {i: [] for i in boxes}
        n_def = n_cln = 0
        for k, j in enumerate(pick):
            i, (x1, x2, y1, y2) = flat[j]
            I = imgs[i]
            side = int(np.clip(round(2 * max(x2 - x1, y2 - y1)), 48, 224)); side = min(side, H, W)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            x0 = int(np.clip(round(cx - side / 2), 0, W - side)); y0 = int(np.clip(round(cy - side / 2), 0, H - side))
            P = I[y0:y0 + side, x0:x0 + side]
            md, sd, mf = float(P.mean()), float(P.std()), material(P)
            fn = f"{s}_{i:04d}_d{k:02d}.png"
            Image.fromarray(P).save(os.path.join(OUT, "img", fn))
            rows.append(dict(file=fn, label=1, series=s, group=GROUP[s], image=i, x0=x0, y0=y0, side=side,
                             bx1=x1, bx2=x2, by1=y1, by2=y2, mean=md, std=sd, material=mf))
            n_def += 1
            # matched clean window
            cand = []
            ids = list(boxes)
            for t in range(1500):
                ii = ids[rng.integers(len(ids))]
                xx, yy = int(rng.integers(0, W - side + 1)), int(rng.integers(0, H - side + 1))
                if overlaps(xx, yy, side, boxes[ii], MARGIN):
                    continue
                if any(iou((xx, yy), u, side) > 0.3 for u in used[ii]):
                    continue
                Q = imgs[ii][yy:yy + side, xx:xx + side]
                qm = material(Q)
                if qm < min(0.8, mf - 0.1):
                    continue
                cand.append((abs(Q.mean() - md) / 32 + abs(Q.std() - sd) / 16, ii, xx, yy, qm))
                if len(cand) >= 300:
                    break
            if not cand:
                log.append(f"{s} img {i}: no clean window found"); continue
            _, ii, xx, yy, qm = min(cand)
            used[ii].append((xx, yy))
            Q = imgs[ii][yy:yy + side, xx:xx + side]
            fn = f"{s}_{ii:04d}_c{k:02d}.png"
            Image.fromarray(Q).save(os.path.join(OUT, "img", fn))
            rows.append(dict(file=fn, label=0, series=s, group=GROUP[s], image=ii, x0=xx, y0=yy, side=side,
                             bx1=np.nan, bx2=np.nan, by1=np.nan, by2=np.nan, mean=float(Q.mean()), std=float(Q.std()), material=qm))
            n_cln += 1
        print(f"{s} [{GROUP[s]}] boxes {len(flat)} -> defective {n_def}, clean {n_cln}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "casting_patches.csv"), index=False)
    print("\n".join(log))
    print(f"TOTAL {len(df)} patches | defective {int(df.label.sum())} | clean {int((df.label == 0).sum())} | "
          f"series {df.series.nunique()} | groups {df.group.nunique()}")
    print(df.groupby("group").label.agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
