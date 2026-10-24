"""HOG + SVM and HOG + RF on W0003, trained on real sections with and without cut-paste defects
(paper: HOG rows of the weld augmentation table and the HOG paired comparisons).

Uses the data loading, defect extraction, pasting and the 25 seeded radiograph-grouped splits of
cutpaste_experiment.py, so the test sets are identical to the ResNet-18 runs. Cut-paste defects are
taken only from W0001 films whose W0003 parent is in the fitting set of the split. Runs on CPU.
Output: hog_cutpaste_perfold.csv (one row per split, regime and model; see results/welds/augmentation/).

Usage:
  python hog_cutpaste.py --w3 <folder with the W0003 PNGs> --labels data/welds/W0003_cell_labels.csv \
                         --w01 <folder containing W0001/ and W0002/>
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutpaste_experiment as cp

ap = argparse.ArgumentParser()
ap.add_argument("--w3", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--w01", required=True)
ap.add_argument("--out", default="hog_cutpaste_perfold.csv")
ap.add_argument("--max-splits", type=int, default=25)
a = ap.parse_args()
cp.W3_DIR, cp.LABELS, cp.REPEATS = a.w3, a.labels, 5
SEED, N_CP = cp.SEED, cp.N_CP

H = lambda x: hog(cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA) / 255.0, orientations=9,
                  pixels_per_cell=(16, 16), cells_per_block=(2, 2), block_norm="L2-Hys")
Xr, Nr, yr, gr = cp.load_real()
Fr = np.array([H(x) for x in Xr], np.float32)
DEF = cp.extract_defects(os.path.join(a.w01, "W0001"), os.path.join(a.w01, "W0002"))
print("real", len(yr), "defects", len(DEF), flush=True)

def best_thr(y, s):
    ts = np.unique(s); ba = [balanced_accuracy_score(y, (s >= t).astype(int)) for t in ts]; return ts[int(np.argmax(ba))]

models = {"SVM": lambda: SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
          "RF": lambda: RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=SEED)}
score = lambda m, X: m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]
rows = []; t0 = time.time()
for n_split, (rep, k, fit, val, te) in enumerate(cp.splits(gr)):
    if n_split >= a.max_splits: break
    rng = np.random.default_rng(SEED + 100 * rep + k)
    blocked = set(gr[val]) | set(gr[te]); D = [d for d in DEF if d["film"] not in blocked]
    src_i = rng.choice(fit[yr[fit] == 0], N_CP)
    Fc = np.array([H(cp.paste(Nr[i], D, rng)) for i in src_i], np.float32)
    sets = {"real": (Fr[fit], yr[fit]), "real+cutpaste": (np.vstack([Fr[fit], Fc]), np.concatenate([yr[fit], np.ones(N_CP, int)]))}
    for reg, (X, y) in sets.items():
        sc = StandardScaler().fit(X)
        for mn, mk in models.items():
            m = mk().fit(sc.transform(X), y)
            thr = best_thr(yr[val], score(m, sc.transform(Fr[val]))); s = score(m, sc.transform(Fr[te]))
            rows.append(dict(rep=rep, fold=k, regime=reg, model=mn, auc=roc_auc_score(yr[te], s),
                             bal_acc_cal=balanced_accuracy_score(yr[te], (s >= thr).astype(int))))
    print(rep, k, f"{time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
print(df.groupby(["model", "regime"])[["auc", "bal_acc_cal"]].agg(["mean", "std"]).round(3))
for mn in models:
    p = df[df.model == mn].pivot_table(index=["rep", "fold"], columns="regime", values="auc")
    d = p["real+cutpaste"] - p["real"]
    print(f"{mn} cut-paste AUC diff {d.mean():+.3f} ± {d.std():.3f} | better in {(d>0).sum()}/{len(d)}")
