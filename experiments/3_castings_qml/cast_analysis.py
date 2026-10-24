# castings analysis (run locally): python3 cast_analysis.py <dir with heads_perfold.csv, bw_sweep.csv, cnn_head_perfold.csv> <out dir>
# %%
import os, sys, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest, wilcoxon
from scipy import stats

def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else None

IN, OUT = sys.argv[1], sys.argv[2]
H = pd.read_csv(os.path.join(IN, "heads_perfold.csv"))
SW = pd.read_csv(os.path.join(IN, "bw_sweep.csv"))
CNN = pd.read_csv(os.path.join(IN, "cnn_head_perfold.csv")) if os.path.exists(os.path.join(IN, "cnn_head_perfold.csv")) else None
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figures", exist_ok=True)
N_SPLITS = H.groupby(["repeat", "fold"]).ngroups
print("splits:", N_SPLITS, "| rows:", len(H))

SRC = {"hog": "HOG", "frozen": "ResNet-18 frozen", "cnnreal": "ResNet-18 fine-tuned"}
MOD = {"svm_rbf": "SVM (RBF)", "svm_lin": "SVM (linear)", "rf": "Random forest",
       "qsvm_zz": "QSVM (ZZ map)", "qsvm_prod": "QSVM (no entanglement)", "pqk": "Projected quantum kernel"}
QUANTUM = ["qsvm_zz", "qsvm_prod", "pqk"]
KEY = ["repeat", "fold"]

def ms(x): return f"{np.mean(x):.3f} ± {np.std(x, ddof=1):.3f}"

def to_latex(df, path, caption, label):
    cols = list(df.columns)
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{" + "l" * (len(df.index.names) if isinstance(df.index, pd.MultiIndex) else 1) + "c" * len(cols) + "}", "\\toprule"]
    idx_names = [n or "" for n in (df.index.names if isinstance(df.index, pd.MultiIndex) else [df.index.name])]
    esc = lambda t: str(t).replace("±", "$\\pm$").replace("_", " ").replace("%", "\\%")
    lines.append(" & ".join(esc(c) for c in idx_names + cols) + " \\\\")
    lines.append("\\midrule")
    for ix, row in df.iterrows():
        ix = ix if isinstance(ix, tuple) else (ix,)
        lines.append(" & ".join(esc(v) for v in list(ix) + list(row.values)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); m = len(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

def paired(a, b):
    d = (a - b).dropna(); nz = d[d != 0]
    wins = int((d > 0).sum())
    p_sign = binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0
    p_wil = wilcoxon(d).pvalue if len(nz) else 1.0
    return dict(mean_diff=d.mean(), wins=f"{wins}/{len(d)}", p_sign=p_sign, p_wilcoxon=p_wil)

# %% [markdown]
# ## Table 1 — all heads, qubit count chosen on validation

# %%
pca = H[H.d > 0]
tuned = pca.loc[pca.groupby(KEY + ["source", "head_train", "model"]).val_auc.idxmax()]
full = H[(H.d == 0) & (H.model == "svm_rbf")].assign(model="svm_rbf_full")
t1_rows = []
for s in SRC:
    for m in list(MOD) + ["svm_rbf_full"]:
        g = (full if m == "svm_rbf_full" else tuned)
        g = g[(g.source == s) & (g.model == m) & (g.head_train == "real")]
        if len(g): t1_rows.append(dict(features=SRC[s], head=MOD.get(m, "SVM (RBF), all features, no PCA"),
                                       AUC=ms(g.auc), BA=ms(g.bal_acc), **{"defect recall": ms(g.rec_def), "clean recall": ms(g.rec_clean)},
                                       **{"median d": int(np.median(g.d)) if m != "svm_rbf_full" else "-"}))
if CNN is not None:
    for m, lab in [("cnn_real", "ResNet-18 fine-tuned")]:
        g = CNN[CNN.model == m]
        t1_rows.append(dict(features=lab, head="CNN's own classifier (end to end)", AUC=ms(g.auc), BA=ms(g.bal_acc),
                            **{"defect recall": ms(g.rec_def), "clean recall": ms(g.rec_clean), "median d": "-"}))
T1 = pd.DataFrame(t1_rows).set_index(["features", "head"])
T1.to_csv(f"{OUT}/tables/table1_all_heads.csv")
to_latex(T1, f"{OUT}/tables/table1_all_heads.tex",
         f"Casting patch classification on held-out parts ({N_SPLITS} part-grouped splits, mean $\\pm$ std). Number of PCA components / qubits chosen per split on the validation parts.",
         "tab:qml_all_heads")

# %% [markdown]
# ## Table 2 — test AUC at each fixed number of qubits

# %%
fixed = pca[pca.head_train == "real"]
T2 = fixed.pivot_table(index=["source", "d"], columns="model", values="auc", aggfunc=lambda x: ms(x))
T2 = T2[[m for m in MOD if m in T2.columns]].rename(columns=MOD).rename(index=SRC, level=0)
T2.to_csv(f"{OUT}/tables/table2_auc_by_qubits.csv")
to_latex(T2, f"{OUT}/tables/table2_auc_by_qubits.tex", "Test AUC (mean $\\pm$ std) at a fixed number of PCA components, equal to the number of qubits for the quantum heads.", "tab:qml_by_qubits")

# %% [markdown]
# ## Table 3 — paired: each quantum head minus SVM (RBF), same split and features

# %%
def paired_table(frame, by):
    out = []
    for keys, g in frame.groupby(by):
        piv = g.pivot_table(index=KEY, columns="model", values="auc")
        for q in QUANTUM:
            if q in piv and "svm_rbf" in piv:
                out.append(dict(zip(by, keys if isinstance(keys, tuple) else (keys,)), head=MOD[q], **paired(piv[q], piv["svm_rbf"])))
    t = pd.DataFrame(out)
    t["p_holm"] = holm(t.p_wilcoxon)
    return t

T3 = paired_table(tuned[tuned.head_train == "real"], ["source"])
T3["source"] = T3.source.map(SRC)
T3b = paired_table(fixed, ["source", "d"]); T3b["source"] = T3b.source.map(SRC)
fmt = lambda t: t.assign(mean_diff=t.mean_diff.map("{:+.3f}".format), p_sign=t.p_sign.map("{:.3g}".format),
                         p_wilcoxon=t.p_wilcoxon.map("{:.3g}".format), p_holm=t.p_holm.map("{:.3g}".format)).rename(
                         columns={"mean_diff": "$\\Delta$AUC", "p_sign": "p (sign)", "p_wilcoxon": "p (Wilcoxon)", "p_holm": "p (Holm)"})
T3.to_csv(f"{OUT}/tables/table3_paired_tuned_d.csv", index=False); T3b.to_csv(f"{OUT}/tables/table3b_paired_each_d.csv", index=False)
to_latex(fmt(T3).set_index(["source", "head"]), f"{OUT}/tables/table3_paired.tex",
         "Paired comparison with the classical RBF SVM on identical splits and features (test AUC, quantum minus classical). Wins: splits where the quantum head is better. p-values: two-sided sign test and Wilcoxon signed-rank; Holm-adjusted over the table.",
         "tab:qml_paired")

# %% [markdown]
# ## Table 5 — kernel diagnostics (selected configurations, real-only training)

# %%
T5 = fixed[fixed.model.isin(["svm_rbf"] + QUANTUM)].groupby(["source", "d", "model"]).agg(
    bandwidth=("bw", "median"), offdiag_mean=("off_mean", "mean"), offdiag_std=("off_std", "mean"),
    KTA=("kta", "mean"), geo_vs_rbf=("geo_vs_rbf", "mean")).round(3)
T5 = T5.rename(index=SRC, level=0).rename(index=MOD, level=2)
T5.to_csv(f"{OUT}/tables/table5_diagnostics.csv")

# %% [markdown]
# ## Figures

# %%
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b", "figure.facecolor": "white", "axes.facecolor": "white",
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 300})
COL = {"svm_rbf": "#2a78d6", "qsvm_zz": "#eb6834", "qsvm_prod": "#1baf7a", "pqk": "#eda100"}
MRK = {"svm_rbf": "o", "qsvm_zz": "s", "qsvm_prod": "^", "pqk": "D"}
GRID = "#e4e3df"

def save(fig, name):
    fig.savefig(f"{OUT}/figures/{name}.pdf"); fig.savefig(f"{OUT}/figures/{name}.png"); plt.close(fig)

# Figure A: AUC vs number of qubits, one panel per feature set
srcs = [s for s in SRC if s in set(fixed.source)]
SHORT = {"hog": "HOG", "frozen": "ResNet-18\nfrozen", "cnnreal": "ResNet-18\nfine-tuned"}
fig, axes = plt.subplots(1, len(srcs), figsize=(2.4 * len(srcs), 2.6), sharey=True)
axes = np.atleast_1d(axes)
for ax, s in zip(axes, srcs):
    g = fixed[fixed.source == s]
    for j, m in enumerate(COL):
        st = g[g.model == m].groupby("d").auc.agg(["mean", "std", "count"])
        x = st.index.values + (j - 1.5) * 0.18
        ax.errorbar(x, st["mean"], yerr=st["std"] / np.sqrt(st["count"]), color=COL[m], marker=MRK[m], ms=4,
                    lw=1.4, elinewidth=0.8, capsize=0, mec="white", mew=0.8, label=MOD[m])
    if CNN is not None and s == "cnnreal":
        ref = CNN[CNN.model == "cnn_real"].auc.mean()
        ax.axhline(ref, color="#52514e", lw=0.9, ls=(0, (4, 3)))
    ax.set_title(SHORT[s], fontsize=8); ax.set_xticks(sorted(g.d.unique()))
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel("test AUC (mean ± s.e.)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COL[m], marker=MRK[m], ms=4, lw=1.4, mec="white") for m in COL]
labels = [MOD[m] for m in COL]
if CNN is not None:
    handles.append(Line2D([], [], color="#52514e", lw=0.9, ls=(0, (4, 3)))); labels.append("CNN's own classifier")
fig.supxlabel("number of PCA components = qubits", fontsize=8, y=0.1)
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.06), handlelength=1.8)
fig.subplots_adjust(bottom=0.26, wspace=0.12)
save(fig, "figA_auc_vs_qubits")

# Figure B: bandwidth sweep for the ZZ QSVM — AUC (left) and kernel concentration (right), main feature set
main_src = "cnnreal"
z = SW[(SW.source == main_src) & (SW.model == "qsvm_zz")]
dims = sorted(z.d.unique()); ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"][-len(dims):]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.4))
for d, c in zip(dims, ramp):
    st = z[z.d == d].groupby("bw").agg(val=("val_auc", "mean"), off=("off_std", "mean"))
    a1.plot(st.index, st.val, color=c, lw=1.6, marker="o", ms=3, mec="white", mew=0.6, label=f"{d} qubits")
    a2.plot(st.index, st.off, color=c, lw=1.6, marker="o", ms=3, mec="white", mew=0.6, label=f"{d} qubits")
for a, t, yl in [(a1, "Validation AUC vs bandwidth", "validation AUC"), (a2, "Kernel spread vs bandwidth", "std of off-diagonal kernel values")]:
    a.set_xscale("log"); a.set_xlabel("bandwidth c"); a.set_ylabel(yl)
    a.set_title(t, fontsize=8); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
a2.legend(fontsize=7, loc="upper right")
fig.suptitle(f"QSVM (ZZ map) on {SRC[main_src]} features", fontsize=8, y=1.02)
fig.subplots_adjust(wspace=0.35)
save(fig, "figB_bandwidth_sweep")

# Figure C: per-split paired scatter, QSVM (ZZ) vs SVM (RBF), qubits tuned on validation
piv = tuned[(tuned.source == main_src) & (tuned.head_train == "real")].pivot_table(index=KEY, columns="model", values="auc")
fig, ax = plt.subplots(figsize=(2.8, 2.8))
lo, hi = np.nanmin(piv[["svm_rbf", "qsvm_zz"]].values) - 0.03, min(1.0, np.nanmax(piv[["svm_rbf", "qsvm_zz"]].values) + 0.03)
ax.plot([lo, hi], [lo, hi], color="#8a8984", lw=0.8)
ax.scatter(piv["svm_rbf"], piv["qsvm_zz"], s=26, color=COL["qsvm_zz"], edgecolor="white", linewidth=0.8, zorder=3)
w = int((piv["qsvm_zz"] > piv["svm_rbf"]).sum())
ax.text(lo + 0.01, hi - 0.01, f"QSVM better in {w}/{len(piv)} splits", va="top", fontsize=7, color="#52514e")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("SVM (RBF) test AUC"); ax.set_ylabel("QSVM (ZZ map) test AUC"); ax.set_title(SHORT[main_src].replace("\n", " "), fontsize=8)
ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
save(fig, "figC_paired_scatter")
print("figures:", sorted(os.listdir(f"{OUT}/figures")))
print("tables:", sorted(os.listdir(f"{OUT}/tables")))


# %% [markdown]
# ## Table 6 — corrected paired test with 95% confidence intervals

# %%
def corrected(a, b, folds=5):
    d = (a - b).dropna(); n = len(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1 / n + 1 / (folds - 1)) * v); t = m / se if se > 0 else 0.0
    h = stats.t.ppf(0.975, n - 1) * se
    return dict(diff=m, ci_lo=m - h, ci_hi=m + h, wins=f"{int((d > 0).sum())}/{n}", p_corr=2 * stats.t.sf(abs(t), n - 1))

rows = []
for ht in ["real"]:
    for s in SRC:
        p = tuned[(tuned.source == s) & (tuned.head_train == ht)].pivot_table(index=KEY, columns="model", values="auc")
        for q in QUANTUM:
            rows.append(dict(head_train=ht, features=SRC[s], comparison=f"{MOD[q]} − SVM (RBF)", **corrected(p[q], p["svm_rbf"])))
        rows.append(dict(head_train=ht, features=SRC[s], comparison="QSVM (ZZ) − QSVM (no entanglement)", **corrected(p["qsvm_zz"], p["qsvm_prod"])))
T6 = pd.DataFrame(rows)
for ht in ["real"]:
    i = T6.head_train == ht; T6.loc[i, "p_holm"] = holm(T6.loc[i, "p_corr"].values)
T6.to_csv(f"{OUT}/tables/table6_corrected_paired.csv", index=False)
f6 = T6.assign(**{"ΔAUC [95% CI]": T6.apply(lambda r: f"{r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", axis=1),
                  "p (corrected)": T6.p_corr.map("{:.2f}".format)})[["head_train", "features", "comparison", "ΔAUC [95% CI]", "wins", "p (corrected)"]]
to_latex(f6[f6.head_train == "real"].drop(columns="head_train").set_index(["features", "comparison"]), f"{OUT}/tables/table6_corrected_paired.tex",
         "Castings: paired test-AUC differences with 95\\% confidence intervals from the corrected repeated cross-validation t-test.",
         "tab:qml_corrected")

# %% [markdown]
# ## Table 8 — what happens without bandwidth tuning

# %%
sw = SW[SW.model.isin(QUANTUM)]
c1 = sw.bw.unique()[np.argmin(np.abs(np.log(sw.bw.unique())))]
at1 = sw[np.isclose(sw.bw, c1)]
at1 = at1.loc[at1.groupby(KEY + ["source", "model"]).val_auc.idxmax()]
tr = tuned[(tuned.head_train == "real") & tuned.model.isin(QUANTUM)]
T8 = pd.concat([at1.groupby(["source", "model"]).auc.agg(lambda x: ms(x)).rename(f"untuned (c = {c1:.2f})"),
                tr.groupby(["source", "model"]).auc.agg(lambda x: ms(x)).rename("bandwidth tuned")], axis=1)
T8 = T8.rename(index=SRC, level=0).rename(index=MOD, level=1)
T8.to_csv(f"{OUT}/tables/table8_untuned_bandwidth.csv")
to_latex(T8, f"{OUT}/tables/table8_untuned_bandwidth.tex",
         f"Test AUC of the quantum heads with features encoded at their standardised scale (bandwidth $c = {c1:.2f}$) versus with the bandwidth tuned on validation parts.",
         "tab:qml_untuned")
