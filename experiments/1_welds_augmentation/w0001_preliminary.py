"""Preliminary weld study on GDXray+ W0001 (paper: Table "Preliminary study on W0001 patches").

HOG + SVM / RF / k-NN on 128x128 patches of the ten W0001 radiographs, labelled from the
W0002 masks, 5-fold cross-validation grouped by radiograph, real test patches only.

Two experiments, both reported as balanced accuracy (mean +- s.d. over folds):
  sd           Real training patches + 0 / 500 / 1,000 / 3,000 Stable Diffusion patches per class,
               and a synthetic-only model ("SD only").
  conventional Real training patches + flips, small rotations, contrast and noise
               ("conventional transforms x4" = 3 augmented copies of every balanced real patch).

This script is the notebook cells that produced the reported numbers (originally run on Kaggle),
collected into one file without changing their logic. The Stable Diffusion images are not part of
GDXray+ and are not redistributed here; the `sd` experiment needs them in the layout
<root>/W0001/*.png (images) and <root>/W0002/*.png (masks, defective set only).

Usage:
  python w0001_preliminary.py --welds <dir containing W0001/ and W0002/> --experiment conventional
  python w0001_preliminary.py --welds ... --synth-def <dir> --synth-clean <dir> --experiment sd
"""
import argparse, os, glob, random
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import rotate
from skimage.feature import hog
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--welds", default="/kaggle/input/gdxray-without-synthetic-for-defectd/Welds kaggle/Welds")
ap.add_argument("--synth-def", default="/kaggle/input/welds-synthetic-generated-img-with-masks/Welds_Synthetic_Defect")
ap.add_argument("--synth-clean", default="/kaggle/input/welds-syntheticnondefect-generated-img-with-masks/Welds_Synthetic_NonDefective")
ap.add_argument("--experiment", choices=["sd", "conventional", "both"], default="both")
ap.add_argument("--out", default=".")
args = ap.parse_args()
REAL_WELDS, SYNTH_DEF, SYNTH_NONDEF = args.welds, args.synth_def, args.synth_clean

# ---- Step 0: settings -------------------------------------------------------------------------
PATCH, STRIDE = 128, 64
MIN_DEF_PIX = 30          # >= 30 mask pixels: defective; 1-29: ambiguous, skipped; 0: clean
MIN_STD = 0.02            # near-uniform background patches are skipped
N_FOLDS = 5
MAX_PER_CLASS = 3000
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# ---- Step 1: helpers --------------------------------------------------------------------------
def mask_path(base, img_path):
    p = os.path.join(base, "W0002", os.path.basename(img_path).replace("W0001", "W0002"))
    return p if os.path.exists(p) else None

def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0

def load_mask(p, shape):
    m = Image.open(p).convert("L")
    if m.size != (shape[1], shape[0]): m = m.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(m) > 0

def feat(p):
    return hog(p, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")

def to_patches(img, mask=None, label=None):
    H, W = img.shape
    if max(H, W) <= 2 * PATCH:
        im = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((PATCH, PATCH)), dtype=np.float32) / 255.0
        lab = label if mask is None else int(mask.sum() >= MIN_DEF_PIX)
        return [feat(im)], [lab]
    X, y = [], []
    for r in range(0, H - PATCH + 1, STRIDE):
        for c in range(0, W - PATCH + 1, STRIDE):
            p = img[r:r + PATCH, c:c + PATCH]
            if p.std() < MIN_STD: continue
            if mask is not None:
                d = mask[r:r + PATCH, c:c + PATCH].sum()
                if 0 < d < MIN_DEF_PIX: continue
                lab = int(d >= MIN_DEF_PIX)
            else:
                lab = label
            X.append(feat(p)); y.append(lab)
    return X, y

models = {
    "SVM": lambda: SVC(kernel="rbf", gamma="scale", random_state=SEED),
    "RF":  lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    "KNN": lambda: KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
}

def report(rows, pooled, name):
    """Step 5: per-fold mean +- s.d. (reported in the paper) and pooled metrics."""
    per_fold = pd.DataFrame(rows)
    print(f"\n=== {name}: PER-FOLD MEAN ± STD (real weld test patches only) ===")
    print(per_fold.groupby(["model", "run"])[["bal_acc", "macro_f1"]].agg(["mean", "std"]).round(3))
    out = []
    for (r, m), (yt, yp) in pooled.items():
        yt, yp = np.array(yt), np.array(yp)
        rec = recall_score(yt, yp, average=None, labels=[0, 1])
        out.append(dict(model=m, run=r, bal_acc=balanced_accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                        recall_nondef=rec[0], recall_def=rec[1]))
        print(f"\n{m} | {r}\n{confusion_matrix(yt, yp, labels=[0, 1])}")
    summary = pd.DataFrame(out).sort_values(["model", "run"]).round(3)
    print(f"\n=== {name}: POOLED OVER ALL FOLDS ===")
    print(summary.to_string(index=False))
    per_fold.to_csv(os.path.join(args.out, f"w0001_{name}_perfold.csv"), index=False)
    summary.to_csv(os.path.join(args.out, f"w0001_{name}_pooled.csv"), index=False)

# ---- Step 2: real W0001 patches (testing and real training) -------------------------------------
Xr, yr, gr = [], [], []
for i, p in enumerate(sorted(glob.glob(os.path.join(REAL_WELDS, "W0001", "*.png")))):
    img = load_gray(p)
    mp = mask_path(REAL_WELDS, p)
    if mp is None: print("no mask for", os.path.basename(p))
    m = load_mask(mp, img.shape) if mp else np.zeros(img.shape, bool)
    X, y = to_patches(img, m)
    Xr += X; yr += y; gr += [i] * len(y)
Xr, yr, gr = np.array(Xr), np.array(yr), np.array(gr)
print("REAL TOTAL:", len(yr), "| defective:", yr.sum(), "| non-defective:", (yr == 0).sum())   # 5,001 | 1,335 | 3,666

# ---- Experiment 1: Stable Diffusion dose --------------------------------------------------------
if args.experiment in ("sd", "both"):
    # Step 3: synthetic patches, at most 3 per class from each generated image
    PER_LABEL = 3
    srng = np.random.default_rng(SEED)

    def load_synth(base, label):
        X, y = [], []
        paths = sorted(glob.glob(os.path.join(base, "W0001", "*.png")))
        for k, p in enumerate(paths):
            img = load_gray(p)
            mp = mask_path(base, p) if label == 1 else None
            m = load_mask(mp, img.shape) if mp else None
            a, b = to_patches(img, m, label)
            a, b = np.array(a, dtype=np.float32), np.array(b)
            for lab in (0, 1):
                idx = np.where(b == lab)[0]
                if len(idx) == 0: continue
                idx = srng.choice(idx, min(PER_LABEL, len(idx)), replace=False)
                X.append(a[idx]); y.append(b[idx])
            if k % 500 == 0: print(f"  {base.split('/')[-1]}: {k}/{len(paths)}")
        return np.vstack(X), np.concatenate(y)

    Xd, yd = load_synth(SYNTH_DEF, 1)
    Xn, yn = load_synth(SYNTH_NONDEF, 0)
    Xs, ys = np.vstack([Xd, Xn]), np.concatenate([yd, yn])
    del Xd, Xn
    print("SYNTH TOTAL:", len(ys), "| defective:", ys.sum(), "| non-defective:", (ys == 0).sum())

    def balance(X, y, rng):
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        n = min(len(i0), len(i1), MAX_PER_CLASS)
        idx = np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])
        return X[idx], y[idx]

    def take(X, y, n, rng):
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        idx = np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])
        return X[idx], y[idx]

    SYN_LEVELS = [0, 500, 1000, 3000]
    runs = [f"real+syn{k}" for k in SYN_LEVELS] + ["C_synth"]     # C_synth = "SD only"
    pooled = {(r, m): ([], []) for r in runs for m in models}
    rows = []
    rng = np.random.default_rng(SEED)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(Xr, yr, gr)):
        Xte, yte = Xr[te], yr[te]
        Xrb, yrb = balance(Xr[tr], yr[tr], rng)
        print(f"\nFold {f} | test radiographs {np.unique(gr[te])} | real train per class: {len(yrb)//2}")
        train_sets = {}
        for k in SYN_LEVELS:
            if k == 0: train_sets["real+syn0"] = (Xrb, yrb)
            else:
                Xsk, ysk = take(Xs, ys, k, rng)
                train_sets[f"real+syn{k}"] = (np.vstack([Xrb, Xsk]), np.concatenate([yrb, ysk]))
        train_sets["C_synth"] = take(Xs, ys, MAX_PER_CLASS, rng)
        for r, (Xb, yb) in train_sets.items():
            sc = StandardScaler().fit(Xb)
            Xb_s, Xte_s = sc.transform(Xb), sc.transform(Xte)
            for m, make in models.items():
                yp = make().fit(Xb_s, yb).predict(Xte_s)
                pooled[(r, m)][0].extend(yte); pooled[(r, m)][1].extend(yp)
                rows.append(dict(fold=f, run=r, model=m, bal_acc=balanced_accuracy_score(yte, yp),
                                 macro_f1=f1_score(yte, yp, average="macro")))
            print(f"  {r}: {len(yb)} samples")
    report(rows, pooled, "sd")

# ---- Experiment 2: conventional augmentation of real patches -----------------------------------
if args.experiment in ("conventional", "both"):
    def raw_patches(img, mask):
        P, y = [], []
        H, W = img.shape
        for r in range(0, H - PATCH + 1, STRIDE):
            for c in range(0, W - PATCH + 1, STRIDE):
                p = img[r:r + PATCH, c:c + PATCH]
                if p.std() < MIN_STD: continue
                d = mask[r:r + PATCH, c:c + PATCH].sum()
                if 0 < d < MIN_DEF_PIX: continue
                P.append((p * 255).astype(np.uint8)); y.append(int(d >= MIN_DEF_PIX))
        return P, y

    Pr, yr2, gr2 = [], [], []
    for i, p in enumerate(sorted(glob.glob(os.path.join(REAL_WELDS, "W0001", "*.png")))):
        img = load_gray(p); m = load_mask(mask_path(REAL_WELDS, p), img.shape)
        P, y = raw_patches(img, m)
        Pr += P; yr2 += y; gr2 += [i] * len(y)
    Pr, yr2, gr2 = np.array(Pr), np.array(yr2), np.array(gr2)
    Fr = np.array([feat(p / 255.0) for p in Pr], dtype=np.float32)
    print("real patches:", len(yr2), "| def:", yr2.sum(), "| nondef:", (yr2 == 0).sum())

    def augment(p, rng):
        x = p.astype(np.float32) / 255.0
        if rng.random() < 0.5: x = x[:, ::-1]
        if rng.random() < 0.5: x = x[::-1, :]
        x = rotate(x, rng.uniform(-8, 8), reshape=False, mode="reflect")
        mu = x.mean()
        x = (x - mu) * rng.uniform(0.85, 1.15) + mu + rng.uniform(-0.06, 0.06)
        x = x + rng.normal(0, rng.uniform(0, 0.02), x.shape)
        return np.clip(x, 0, 1)

    def aug_set(P, y, copies, rng):
        F = np.array([feat(augment(p, rng)) for p in P for _ in range(copies)], dtype=np.float32)
        return F, np.repeat(y, copies)

    def bal_idx(y, rng):
        i0, i1 = np.where(y == 0)[0], np.where(y == 1)[0]
        n = min(len(i0), len(i1))
        return np.concatenate([rng.choice(i0, n, replace=False), rng.choice(i1, n, replace=False)])

    runs = ["base", "aug_x2", "aug_x4", "all+augdef"]     # aug_x4 = "conventional transforms (x4)"
    pooled = {(r, m): ([], []) for r in runs for m in models}
    rows = []
    rng = np.random.default_rng(SEED)
    for f, (tr, te) in enumerate(GroupKFold(n_splits=N_FOLDS).split(Fr, yr2, gr2)):
        Xte, yte = Fr[te], yr2[te]
        bi = tr[bal_idx(yr2[tr], rng)]
        Xb, yb = Fr[bi], yr2[bi]
        A1, a1 = aug_set(Pr[bi], yr2[bi], 1, rng)
        A3, a3 = aug_set(Pr[bi], yr2[bi], 3, rng)
        i0, i1 = tr[yr2[tr] == 0], tr[yr2[tr] == 1]
        need = len(i0) - len(i1)
        src = rng.choice(i1, need, replace=True)
        Ad = np.array([feat(augment(p, rng)) for p in Pr[src]], dtype=np.float32)
        train_sets = {
            "base":       (Xb, yb),
            "aug_x2":     (np.vstack([Xb, A1]), np.concatenate([yb, a1])),
            "aug_x4":     (np.vstack([Xb, A3]), np.concatenate([yb, a3])),
            "all+augdef": (np.vstack([Fr[tr], Ad]), np.concatenate([yr2[tr], np.ones(need, int)])),
        }
        print(f"\nFold {f} | test radiographs {np.unique(gr2[te])}")
        for r, (Xt, yt) in train_sets.items():
            sc = StandardScaler().fit(Xt)
            Xt_s, Xte_s = sc.transform(Xt), sc.transform(Xte)
            for m, make in models.items():
                yp = make().fit(Xt_s, yt).predict(Xte_s)
                pooled[(r, m)][0].extend(yte); pooled[(r, m)][1].extend(yp)
                rows.append(dict(fold=f, run=r, model=m, bal_acc=balanced_accuracy_score(yte, yp),
                                 macro_f1=f1_score(yte, yp, average="macro")))
            print(f"  {r}: {len(yt)} samples | def={int(yt.sum())} nondef={int((yt==0).sum())}")
    report(rows, pooled, "conventional")
