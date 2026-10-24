"""Write the 25 train/validation/test splits used in the paper as CSV files.

Both materials use the same rule: 5 repeats x 5 folds, folds assigned by a seeded permutation of the
groups (seed 42 + repeat), and 20% of the training groups held out for validation. Groups are
radiographs for welds (W0003 film) and physical parts for castings. The rule below is copied from
the notebooks (weld-qml-01, cast-qml-01) and cutpaste_experiment.py, which all generate their
splits with it at run time; this script only saves the result.

Outputs (in data/welds and data/castings):
  *_splits.csv        one row per split and sample: repeat, fold, sample index, role (fit / val / test)
  *_split_groups.csv  one row per split: sizes and the groups in each role
Usage:  python scripts/export_splits.py            (run from the repository root)
"""
import os
import numpy as np, pandas as pd

SEED, REPEATS, FOLDS, VAL_FRAC = 42, 5, 5, 0.2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def splits(groups):
    units = np.array(sorted(set(groups)))
    for rep in range(REPEATS):
        r = np.random.default_rng(SEED + rep)
        perm = r.permutation(units)
        fold_of = {u: i % FOLDS for i, u in enumerate(perm)}
        f = np.array([fold_of[x] for x in groups])
        for k in range(FOLDS):
            tr_units = sorted(set(groups[f != k]))
            val_units = set(r.choice(tr_units, max(1, int(round(VAL_FRAC * len(tr_units)))), replace=False))
            fit = np.where((f != k) & ~np.isin(groups, list(val_units)))[0]
            val = np.where(np.isin(groups, list(val_units)))[0]
            yield rep, k, fit, val, np.where(f == k)[0]

def export(meta, group_col, id_col, out_prefix):
    groups = meta[group_col].values
    long, summ = [], []
    for rep, k, fit, val, te in splits(groups):
        assert not (set(groups[fit]) & set(groups[val]) or set(groups[fit]) & set(groups[te]) or set(groups[val]) & set(groups[te]))
        for role, idx in (("fit", fit), ("val", val), ("test", te)):
            long += [dict(repeat=rep, fold=k, idx=int(i), **{id_col: meta[id_col].iloc[i]}, role=role) for i in idx]
        summ.append(dict(repeat=rep, fold=k, n_fit=len(fit), n_val=len(val), n_test=len(te),
                         fit_groups=";".join(sorted(set(groups[fit]))), val_groups=";".join(sorted(set(groups[val]))),
                         test_groups=";".join(sorted(set(groups[te])))))
    pd.DataFrame(long).to_csv(f"{out_prefix}_splits.csv", index=False)
    pd.DataFrame(summ).to_csv(f"{out_prefix}_split_groups.csv", index=False)
    print(f"{os.path.relpath(out_prefix, ROOT)}: {len(summ)} splits over {len(set(groups))} groups, {len(meta)} samples")

if __name__ == "__main__":
    # welds: 575 sections in data/welds/weld_sections.csv (the order the notebooks load them), grouped by W0003 film
    sec = pd.read_csv(os.path.join(ROOT, "data/welds/weld_sections.csv"))
    sec["section"] = sec.image.str[:-4] + "_c" + sec.cell.astype(str).str.zfill(2)
    export(sec, "image", "section", os.path.join(ROOT, "data/welds/weld"))
    # castings: 1,296 patches in data/castings/casting_patches.csv (file order), grouped by physical part
    export(pd.read_csv(os.path.join(ROOT, "data/castings/casting_patches.csv")), "group", "file", os.path.join(ROOT, "data/castings/casting"))
