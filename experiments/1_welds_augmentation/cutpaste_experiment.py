import os, sys, glob, time, numpy as np, pandas as pd, cv2
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score

W3_DIR    = "/kaggle/input/datasets/samarthgoel2604/gdxray-w0003-labelled/W0003"
LABELS    = "/kaggle/input/datasets/samarthgoel2604/gdxray-w0003-labelled/W0003_cell_labels.csv"
SYN_DEF   = "/kaggle/input/datasets/samarthgoel2604/welds-synthetic-generated-img-with-masks/Welds_Synthetic_Defect"
SYN_CLEAN = "/kaggle/input/datasets/samarthgoel2604/welds-syntheticnondefect-generated-img-with-masks/Welds_Synthetic_NonDefective"
W12_DIR   = "/kaggle/input/datasets/samarthgoel2604/gdxray-without-synthetic-for-defectd/Welds kaggle/Welds"
OUT = "/kaggle/working"

SIZE, SEED = 224, 42
REPEATS, FOLDS = 5, 5   # reported run: 5 repeats = 25 splits (its first 15 splits equal the earlier 3-repeat run)
N_SYN = 1500
STEPS, FT_STEPS, BS, LR = 600, 300, 32, 3e-4
MIN_DEF_PIX = 30
SECTION_RATIO = 0.6
PRETRAINED = True
REGIMES = ["real", "real+cutpaste", "real+SD"]
N_CP = 1500
VAL_FRAC = 0.2

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

W1_TO_W3 = {1: "W0003_0068.png", 2: "W0003_0011.png", 3: "W0003_0030.png", 4: "W0003_0036.png", 5: "W0003_0041.png",
            6: "W0003_0042.png", 7: "W0003_0012.png", 8: "W0003_0039.png", 9: "W0003_0006.png", 10: "W0003_0007.png"}
MIN_AREA, PADD = 15, 8

def extract_defects(w1_dir, w2_dir):
    out = []
    for k, film in W1_TO_W3.items():
        I = cv2.imread(os.path.join(w1_dir, f"W0001_{k:04d}.png"), 0)
        M = cv2.imread(os.path.join(w2_dir, f"W0002_{k:04d}.png"), 0)
        if I is None or M is None: continue
        if M.shape != I.shape: M = cv2.resize(M, (I.shape[1], I.shape[0]), interpolation=cv2.INTER_NEAREST)
        M = (M > 0).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(M, 8)
        If = I.astype(np.float32)
        for c in range(1, n):
            x, y, w, h, a = st[c]
            if a < MIN_AREA or w > 400 or h > 200: continue
            x0, y0 = max(0, x - PADD), max(0, y - PADD)
            x1, y1 = min(I.shape[1], x + w + PADD), min(I.shape[0], y + h + PADD)
            patch = I[y0:y1, x0:x1]
            m = (lab[y0:y1, x0:x1] == c).astype(np.uint8)
            md = cv2.dilate(m, np.ones((5, 5), np.uint8))
            bg = cv2.inpaint(patch, md * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
            delta = (patch.astype(np.float32) - bg) * cv2.GaussianBlur(md.astype(np.float32), (5, 5), 0)
            out.append(dict(film=film, delta=delta, area=int(a)))
    return out

def paste(section, defects, rng, n_max=3):
    S = section.astype(np.float32); H, W = S.shape
    k = rng.integers(1, n_max + 1)
    for d in rng.choice(len(defects), k, replace=False):
        D = defects[d]["delta"]
        if rng.random() < 0.5: D = D[:, ::-1]
        if rng.random() < 0.5: D = D[::-1, :]
        D = D * rng.uniform(0.8, 1.2)
        h, w = D.shape
        if h >= H or w >= W:
            s = min((H - 2) / h, (W - 2) / w); D = cv2.resize(D, (max(1, int(w * s)), max(1, int(h * s)))); h, w = D.shape
        cy = int(rng.uniform(0.3, 0.7) * H); cx = int(rng.uniform(0, W))
        y0 = int(np.clip(cy - h // 2, 0, H - h)); x0 = int(np.clip(cx - w // 2, 0, W - w))
        S[y0:y0 + h, x0:x0 + w] += D
    return np.clip(S, 0, 255).astype(np.uint8)

def load_real():
    lab = pd.read_csv(LABELS)
    lab = lab[lab.clean_margin_ok == 1]
    X, N, y, g = [], [], [], []
    for img, d in lab.groupby("image", sort=True):
        A = cv2.imread(os.path.join(W3_DIR, img), 0)
        if A is None: print("missing", img); continue
        a, b = weld_band(A)
        for _, r in d.iterrows():
            if r.x1 > A.shape[1]: continue
            sec = A[a:b, r.x0:r.x1].copy()
            N.append(sec); X.append(rs(sec)); y.append(int(r.label)); g.append(img)
    return np.array(X, np.uint8), N, np.array(y), np.array(g)

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


def splits(groups):
    imgs = np.array(sorted(set(groups)))
    for rep in range(REPEATS):
        r = np.random.default_rng(SEED + rep)
        perm = r.permutation(imgs)
        fold_of = {im: i % FOLDS for i, im in enumerate(perm)}
        f = np.array([fold_of[x] for x in groups])
        for k in range(FOLDS):
            tr_imgs = sorted(set(groups[f != k]))
            val_imgs = set(r.choice(tr_imgs, max(1, int(round(VAL_FRAC * len(tr_imgs)))), replace=False))
            fit = np.where((f != k) & ~np.isin(groups, list(val_imgs)))[0]
            val = np.where(np.isin(groups, list(val_imgs)))[0]
            yield rep, k, fit, val, np.where(f == k)[0]

def best_threshold(yt, p):
    ts = np.unique(np.concatenate([p, [0.5]]))
    ba = [balanced_accuracy_score(yt, (p >= t).astype(int)) for t in ts]
    return float(ts[int(np.argmax(ba))])

def grid(rows, path):
    out = []
    for t, ims in rows:
        row = np.hstack([cv2.copyMakeBorder(cv2.resize(i, (160, 160)), 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255) for i in ims])
        bar = np.full((34, row.shape[1]), 255, np.uint8)
        cv2.putText(bar, t, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        out += [bar, row]
    cv2.imwrite(path, np.vstack(out))

if __name__ == "__main__":
    import torch, torch.nn as nn, torchvision
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)
    t0 = time.time()
    Xr, Nr, yr, gr = load_real()
    print(f"REAL: {len(yr)} sections | defective {yr.sum()} | clean {(yr==0).sum()} | radiographs {len(set(gr))}", flush=True)
    DEF = extract_defects(os.path.join(W12_DIR, "W0001"), os.path.join(W12_DIR, "W0002"))
    print(f"CUT-PASTE: {len(DEF)} real defects extracted from W0001", flush=True)
    Xs, ys = load_synth(np.random.default_rng(SEED))
    print(f"SD SYNTH: {len(ys)} crops | defective {ys.sum()} | clean {(ys==0).sum()} | {time.time()-t0:.0f}s", flush=True)
    gr_ = np.random.default_rng(0)
    ci = gr_.choice(np.where(yr == 0)[0], 8, replace=False)
    grid([("REAL defective", [Xr[i] for i in gr_.choice(np.where(yr == 1)[0], 8, replace=False)]),
          ("REAL clean (paste target)", [Xr[i] for i in ci]),
          ("CUT-PASTE defective", [rs(paste(Nr[i], DEF, gr_)) for i in ci])], f"{OUT}/cutpaste_samples.png")

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
    for rep, k, fit, val, te in splits(gr):
        rng = np.random.default_rng(SEED + 100 * rep + k)
        blocked = set(gr[val]) | set(gr[te])
        D = [d for d in DEF if d["film"] not in blocked]
        clean_fit = fit[yr[fit] == 0]
        src = rng.choice(clean_fit, N_CP)
        Xc = np.array([rs(paste(Nr[i], D, rng)) for i in src], np.uint8)
        CP = (np.concatenate([Xc, Xr[clean_fit]]), np.concatenate([np.ones(N_CP, int), np.zeros(len(clean_fit), int)]))
        i0, i1 = np.where(ys == 0)[0], np.where(ys == 1)[0]
        si = np.concatenate([rng.choice(i0, min(N_SYN, len(i0)), replace=False), rng.choice(i1, min(N_SYN, len(i1)), replace=False)])
        R, S = (Xr[fit], yr[fit]), (Xs[si], ys[si])
        for reg in REGIMES:
            torch.manual_seed(SEED + 100 * rep + k)
            m = make_model()
            if reg == "real": train(m, [R], STEPS, LR, rng)
            elif reg == "real+cutpaste": train(m, [R, CP], STEPS, LR, rng)
            else: train(m, [R, S], STEPS, LR, rng)
            thr = best_threshold(yr[val], predict(m, Xr[val]))
            p = predict(m, Xr[te]); yt = yr[te]
            row = dict(repeat=rep, fold=k, regime=reg, n_defects_used=len(D), thr=thr,
                       auc=roc_auc_score(yt, p) if len(set(yt)) == 2 else np.nan)
            for name, t in [("cal", thr), ("05", 0.5)]:
                yp = (p >= t).astype(int); rec = recall_score(yt, yp, average=None, labels=[0, 1], zero_division=0)
                row.update({f"bal_acc_{name}": balanced_accuracy_score(yt, yp), f"rec_def_{name}": rec[1], f"rec_clean_{name}": rec[0]})
            rows.append(row)
            print(f"rep {rep} fold {k} | {reg:14s} | AUC {row['auc']:.3f} | BA cal {row['bal_acc_cal']:.3f} (thr {thr:.2f}) "
                  f"| BA@0.5 {row['bal_acc_05']:.3f} | {time.time()-t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/cutpaste_perfold.csv", index=False)

    df = pd.DataFrame(rows)
    cols = ["auc", "bal_acc_cal", "rec_def_cal", "rec_clean_cal", "bal_acc_05"]
    print("\n=== MEAN ± STD over", REPEATS * FOLDS, "splits (test = real W0003 sections only) ===")
    print(df.groupby("regime")[cols].agg(["mean", "std"]).round(3).reindex(REGIMES))
    print("\n=== PAIRED vs real-only (same test split) ===")
    piv = df.pivot_table(index=["repeat", "fold"], columns="regime", values=["auc", "bal_acc_cal"])
    for reg in REGIMES[1:]:
        for met in ["auc", "bal_acc_cal"]:
            d = (piv[(met, reg)] - piv[(met, "real")]).dropna()
            print(f"{reg:14s} {met:12s}: mean diff {d.mean():+.3f} ± {d.std():.3f} | better in {(d > 0).sum()}/{len(d)} splits")
