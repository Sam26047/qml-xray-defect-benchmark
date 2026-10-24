import os, glob, time, numpy as np, pandas as pd, cv2
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score

W3_DIR    = "/kaggle/input/datasets/samarthgoel2604/gdxray-w0003-labelled/W0003"
LABELS    = "/kaggle/input/datasets/samarthgoel2604/gdxray-w0003-labelled/W0003_cell_labels.csv"
SYN_DEF   = "/kaggle/input/datasets/samarthgoel2604/welds-synthetic-generated-img-with-masks/Welds_Synthetic_Defect"
SYN_CLEAN = "/kaggle/input/datasets/samarthgoel2604/welds-syntheticnondefect-generated-img-with-masks/Welds_Synthetic_NonDefective"
OUT = "/kaggle/working"

SIZE, SEED = 224, 42
REPEATS, FOLDS = 3, 5
N_SYN = 1500
STEPS, FT_STEPS, BS, LR = 600, 300, 32, 3e-4
MIN_DEF_PIX = 30
SECTION_RATIO = 0.6
PRETRAINED = True
REGIMES = ["real", "synth", "real+synth", "synth->real"]

def weld_band(A, pad=0.3):
    p = np.median(A.astype(np.float32), axis=1)
    p = cv2.GaussianBlur(p.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=25).ravel()
    lo, hi = np.percentile(p, 10), np.percentile(p, 95)
    on = np.append(p > lo + 0.5 * (hi - lo), False)
    best, start = (0, 0, 0), 0
    for i in range(len(on)):
        if on[i] and (i == 0 or not on[i-1]): start = i
        if not on[i] and i > 0 and on[i-1] and i - start > best[0]: best = (i - start, start, i)
    _, a, b = best; h = b - a
    return max(0, int(a - pad * h)), min(A.shape[0], int(b + pad * h))

def rs(c): return cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def load_real():
    lab = pd.read_csv(LABELS)
    lab = lab[lab.clean_margin_ok == 1]
    X, y, g = [], [], []
    for img, d in lab.groupby("image", sort=True):
        A = cv2.imread(os.path.join(W3_DIR, img), 0)
        if A is None: print("missing", img); continue
        a, b = weld_band(A)
        for _, r in d.iterrows():
            if r.x1 > A.shape[1]: continue
            X.append(rs(A[a:b, r.x0:r.x1])); y.append(int(r.label)); g.append(img)
    return np.array(X, np.uint8), np.array(y), np.array(g)

def load_synth(rng):
    X, y = [], []
    for base, is_def in [(SYN_DEF, True), (SYN_CLEAN, False)]:
        paths = sorted(glob.glob(os.path.join(base, "W0001", "*.png")))
        for k, p in enumerate(paths):
            A = cv2.imread(p, 0)
            a, b = weld_band(A); bh = b - a
            cw = max(32, min(A.shape[1], int(SECTION_RATIO * bh)))
            starts = np.arange(0, A.shape[1] - cw + 1, max(1, cw // 4))
            if is_def:
                M = cv2.imread(os.path.join(base, "W0002", os.path.basename(p)), 0)
                if M is None: continue
                if M.shape != A.shape: M = cv2.resize(M, (A.shape[1], A.shape[0]), interpolation=cv2.INTER_NEAREST)
                col = (M[a:b] > 0).sum(0)
                cs = np.concatenate([[0], np.cumsum(col)])
                win = cs[starts + cw] - cs[starts]
                dpos, cpos = starts[win >= MIN_DEF_PIX], starts[win == 0]
                if len(dpos): x = rng.choice(dpos); X.append(rs(A[a:b, x:x+cw])); y.append(1)
                if len(cpos): x = rng.choice(cpos); X.append(rs(A[a:b, x:x+cw])); y.append(0)
            else:
                x = rng.choice(starts); X.append(rs(A[a:b, x:x+cw])); y.append(0)
            if k % 1000 == 0: print(f"  {os.path.basename(base)}: {k}/{len(paths)}", flush=True)
    return np.array(X, np.uint8), np.array(y)

def sample_grid(Xr, yr, Xs, ys, path, n=8, seed=0):
    r = np.random.default_rng(seed); rows = []
    for X, y in [(Xr, yr), (Xs, ys)]:
        for lab in (1, 0):
            idx = r.choice(np.where(y == lab)[0], n, replace=False)
            rows.append(np.hstack([cv2.copyMakeBorder(cv2.resize(X[i], (160, 160)), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255) for i in idx]))
    titles = ["REAL defective", "REAL clean", "SYNTH defective", "SYNTH clean"]
    out = []
    for t, row in zip(titles, rows):
        bar = np.full((34, row.shape[1]), 255, np.uint8)
        cv2.putText(bar, t, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        out += [bar, row]
    cv2.imwrite(path, np.vstack(out))

def splits(groups):
    imgs = np.array(sorted(set(groups)))
    for rep in range(REPEATS):
        perm = np.random.default_rng(SEED + rep).permutation(imgs)
        fold_of = {im: i % FOLDS for i, im in enumerate(perm)}
        f = np.array([fold_of[x] for x in groups])
        for k in range(FOLDS):
            yield rep, k, np.where(f != k)[0], np.where(f == k)[0]

if __name__ == "__main__":
    import torch, torch.nn as nn, torchvision
    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    t0 = time.time()
    Xr, yr, gr = load_real()
    print(f"REAL: {len(yr)} sections | defective {yr.sum()} | clean {(yr==0).sum()} | radiographs {len(set(gr))}", flush=True)
    Xs, ys = load_synth(np.random.default_rng(SEED))
    print(f"SYNTH: {len(ys)} crops | defective {ys.sum()} | clean {(ys==0).sum()} | {time.time()-t0:.0f}s", flush=True)
    sample_grid(Xr, yr, Xs, ys, f"{OUT}/real_vs_synth_samples.png")

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(xb, train):
        x = torch.from_numpy(xb).to(dev).float().div(255).unsqueeze(1)
        if train:
            n = x.shape[0]
            x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.5, x.flip(3), x)
            x = torch.where(torch.rand(n, 1, 1, 1, device=dev) < 0.5, x.flip(2), x)
            mu = x.mean((2, 3), keepdim=True)
            x = (x - mu) * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.85, 1.15) + mu
            x = x + torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.06, 0.06)
            x = (x + torch.randn_like(x) * 0.01).clamp(0, 1)
        return (x.repeat(1, 3, 1, 1) - MEAN) / STD

    def make_model():
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if PRETRAINED else None
        m = torchvision.models.resnet18(weights=w); m.fc = nn.Linear(512, 1)
        return m.to(dev)

    def train(m, sources, steps, lr, rng):
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        lossf = nn.BCEWithLogitsLoss(); m.train()
        pools = [(X, [np.where(y == 0)[0], np.where(y == 1)[0]], y) for X, y in sources]
        per = BS // (2 * len(pools))
        for _ in range(steps):
            xs, ts = [], []
            for X, (i0, i1), y in pools:
                idx = np.concatenate([rng.choice(i0, per), rng.choice(i1, per)])
                xs.append(X[idx]); ts.append(y[idx])
            xb = prep(np.concatenate(xs), True)
            tb = torch.from_numpy(np.concatenate(ts)).float().to(dev)
            with torch.autocast(device_type="cuda", enabled=dev == "cuda"):
                loss = lossf(m(xb).squeeze(1), tb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return m

    @torch.no_grad()
    def predict(m, X):
        m.eval(); out = []
        for i in range(0, len(X), 64):
            out.append(torch.sigmoid(m(prep(X[i:i+64], False)).squeeze(1)).float().cpu().numpy())
        return np.concatenate(out)

    rows = []
    for rep, k, tr, te in splits(gr):
        rng = np.random.default_rng(SEED + 100 * rep + k)
        i0, i1 = np.where(ys == 0)[0], np.where(ys == 1)[0]
        si = np.concatenate([rng.choice(i0, min(N_SYN, len(i0)), replace=False), rng.choice(i1, min(N_SYN, len(i1)), replace=False)])
        R, S = (Xr[tr], yr[tr]), (Xs[si], ys[si])
        for reg in REGIMES:
            torch.manual_seed(SEED + 100 * rep + k)
            m = make_model()
            if reg == "real": train(m, [R], STEPS, LR, rng)
            elif reg == "synth": train(m, [S], STEPS, LR, rng)
            elif reg == "real+synth": train(m, [R, S], STEPS, LR, rng)
            else: train(m, [S], STEPS, LR, rng); train(m, [R], FT_STEPS, LR / 3, rng)
            p = predict(m, Xr[te]); yp = (p >= 0.5).astype(int); yt = yr[te]
            rec = recall_score(yt, yp, average=None, labels=[0, 1], zero_division=0)
            rows.append(dict(repeat=rep, fold=k, regime=reg, bal_acc=balanced_accuracy_score(yt, yp),
                             auc=roc_auc_score(yt, p) if len(set(yt)) == 2 else np.nan,
                             recall_def=rec[1], recall_clean=rec[0]))
            r = rows[-1]
            print(f"rep {rep} fold {k} | {reg:12s} | BA {r['bal_acc']:.3f} AUC {r['auc']:.3f} "
                  f"def {r['recall_def']:.2f} clean {r['recall_clean']:.2f} | {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/cnn_synth_perfold.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== MEAN ± STD over", REPEATS * FOLDS, "splits (test = real W0003 sections only) ===")
    print(df.groupby("regime")[["bal_acc", "auc", "recall_def", "recall_clean"]].agg(["mean", "std"]).round(3).reindex(REGIMES))
    print("\n=== PAIRED vs real-only (same test split) ===")
    piv = df.pivot_table(index=["repeat", "fold"], columns="regime", values=["bal_acc", "auc"])
    for reg in REGIMES[1:]:
        for met in ["bal_acc", "auc"]:
            d = (piv[(met, reg)] - piv[(met, "real")]).dropna()
            print(f"{reg:12s} {met:7s}: mean diff {d.mean():+.3f} ± {d.std():.3f} | better in {(d > 0).sum()}/{len(d)} splits")
