# Quantum kernels vs. classical ML for X-ray defect detection on GDXray+

Code, labels and split definitions for the paper

> S. Goel, P. Mittal, S. Singh. *Quantum Kernels versus Classical Classifiers for Radiographic Defect Detection in Welds and Castings: A Group-Held-Out Benchmark on GDXray+* (under review).

The benchmark compares quantum-kernel SVMs (ZZ feature map, a version without entanglement, and a projected quantum kernel, all simulated exactly) with classical SVMs and random forests on weld radiographs (GDXray+ W0003) and aluminium castings (GDXray+ C series). Every test set is held out by group: by radiograph for welds and by physical part for castings. The repository also contains the weld augmentation study (Stable Diffusion images vs. cut-paste of real defects).

## Contents

```
data/
  welds/
    W0003_cell_labels.csv     section labels for the 38 annotated W0003 radiographs (679 one-centimetre sections)
    W0003_films.csv           one row per radiograph: original BAM file, round-robin number, BAM specimen label, weld ID
    weld_sections.csv         the 575 sections used in the experiments, in the order the code loads them
    weld_splits.csv           the 25 splits: repeat, fold, section, role (fit / val / test)
    weld_split_groups.csv     the radiographs in each role of each split
  castings/
    casting_patches.csv       the 1,296 casting patches: source image, coordinates, label, part group
    casting_splits.csv        the 25 splits: repeat, fold, patch, role
    casting_split_groups.csv  the parts in each role of each split
scripts/
  make_weld_labels.py         builds W0003_cell_labels.csv from BAM's real-values.xls
  make_casting_patches.py     cuts the casting patches from the GDXray+ castings and writes casting_patches.csv
  export_splits.py            writes the split files above from the seeded rule used by all experiments
experiments/
  1_welds_augmentation/       augmentation study (W0001 preliminary study, ResNet-18 and HOG on W0003)
  2_welds_qml/                weld benchmark, Kaggle notebooks 01-04
  3_castings_qml/             casting benchmark, Kaggle notebooks 01-03 and the analysis script
results/                      per-split outputs of every experiment, from which all tables and figures are computed
figures/                      scripts for the overview and sample figures
```

## Data

The radiographs are not redistributed here. Download GDXray+ from its authors (Mery et al., 2015) and use:

* **Welds:** `W0001` (10 radiographs), `W0002` (their defect masks) and `W0003` (BAM round-robin films with `real-values.xls`).
* **Castings:** the `Castings` series (`C0001` ... `C0087`); the series with `ground_truth.txt` bounding boxes are used.

**Weld labels.** BAM's `real-values.xls` gives the true defects of 38 W0003 films for one-centimetre sections measured from the film's zero mark. `make_weld_labels.py` converts this into section labels (248 px per section at 40.3 µm per pixel). A clean section that borders a defective one is marked `clean_margin_ok = 0` and excluded, which leaves 575 sections (351 defective, 224 clean). The script reproduces `W0003_cell_labels.csv` byte for byte:

```
python scripts/make_weld_labels.py <GDXray>/Welds/W0003 data/welds/W0003_cell_labels.csv
```

**Welds and radiographs.** The splits group sections by radiograph (38 groups), as in the paper. `W0003_films.csv` lists BAM's specimen label for every film. Five films are repeat exposures of two pipe welds, taken from different directions: `SLV273A/B/C` (W0003_0053-0055, 5 sections) and `SLV28UA/UB` (W0003_0061-0062, 24 sections). The other 33 films carry distinct labels; `A481` and `A48` (W0003_0030 and W0003_0042) share the specimen number 48 and may also show one weld. To group by physical weld instead (35 groups), use the `weld` column.

**Casting patches.** `make_casting_patches.py` cuts a square patch centred on each ground-truth box (side twice the box, 48-224 px) and, for each one, a clean patch of the same size from the same series, away from every box and matched in intensity. Each series is one part imaged at several angles, and some parts appear in several series, so series of the same part share a `group` (18 part groups); the script's header documents the choices. It regenerates the patch images named in `casting_patches.csv` from the GDXray+ castings:

```
python scripts/make_casting_patches.py <GDXray>/Castings <output folder>
```

**Synthetic weld images.** The Stable Diffusion images used in the augmentation study (3,000 defective and 1,200 defect-free images with masks) are not part of GDXray+ and are not included. Only the two augmentation experiments that use them need them.

## Splits

All experiments use the same rule: 5 repeats × 5 folds, folds assigned by a seeded permutation of the groups (seed 42 + repeat), and 20% of the training groups held out for validation (used only to choose hyperparameters and decision thresholds). The code generates the splits at run time; `scripts/export_splits.py` saves the same splits as CSV, and they match the split files written by the notebooks exactly.

## Reproducing the results

The experiments were run on Kaggle (GPU T4 for the ResNet-18 trainings, CPU for everything else). The notebooks find their inputs anywhere under `/kaggle/input`, so dataset names do not matter. Each notebook's header lists its inputs, outputs, settings and runtime.

**1. Augmentation study (welds)**, `experiments/1_welds_augmentation/`

| Script | Reported as | Needs |
|---|---|---|
| `w0001_preliminary.py` | preliminary W0001 study (HOG + SVM, SD and conventional augmentation) | W0001, W0002; SD images for `--experiment sd` |
| `cutpaste_experiment.py` | ResNet-18 trained on real, real + cut-paste and real + SD sections (25 splits) | W0001, W0002, W0003, labels, SD images; GPU |
| `hog_cutpaste.py` | HOG + SVM / RF with and without cut-paste, same 25 splits | W0001, W0002, W0003, labels; CPU |
| `cnn_synth_experiment.py` | earlier ResNet-18 run with SD images (15 splits, discussed in the text) | W0003, labels, SD images; GPU |

**2. Weld benchmark**, `experiments/2_welds_qml/`. Run the notebooks in order and add each notebook's output as an input of the next:
`01-cnn-embeddings` (GPU; HOG, frozen and fine-tuned ResNet-18 features for each split) → `02-kernel-heads` (all classical and quantum heads, bandwidth sweep) → `03-analysis` (tables, figures, statistical tests) → `04-learning-curve` (small-data test and the geometric-difference λ sweep).

**3. Casting benchmark**, `experiments/3_castings_qml/`: `cast-qml-01` (GPU) → `cast-qml-02` → `cast-qml-03` (learning curve), then the analysis, which runs locally on the per-split results:

```
python experiments/3_castings_qml/cast_analysis.py results/castings/qml <output folder>
```

The quantum kernels are computed by exact state-vector simulation in NumPy (`zz_states`, `prod_kernel` and `bloch` in notebook 02). Notebook 02 checks them against Qiskit's `zz_feature_map` and `z_feature_map` when Qiskit is installed.

## Results

`results/` holds the per-split outputs that the paper's tables and figures are computed from:

* `welds/augmentation/hog_cutpaste_perfold.csv`: HOG heads of the augmentation study.
* `welds/qml/`, `castings/qml/`: `heads_perfold.csv` (every head, feature set, qubit count and split, with the validation-selected bandwidth and C), `bw_sweep.csv` (every bandwidth), `cnn_head_perfold.csv` (the fine-tuned CNN as a classifier), `learning_curve.csv` and `geo_lambda.csv`.

## Determinism

Fold assignment, validation groups, cut-paste sampling and the classical and quantum heads are seeded. On CPU, `hog_cutpaste.py` reproduces the published per-split AUCs exactly, `w0001_preliminary.py --experiment conventional` reproduces the reported balanced accuracies, and the HOG heads of notebook 02 reproduce them up to a few near-tied bandwidth choices (mean AUCs agree to three decimals). The ResNet-18 trainings run on a GPU with cuDNN, so their results can differ slightly between runs and hardware.

## Requirements

Python 3.10+, `numpy`, `pandas`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python`, `Pillow`, `matplotlib`, `xlrd` (labels only); `torch` and `torchvision` for the ResNet-18 notebooks; `qiskit` (optional, cross-check only). See `requirements.txt`.

## Licence and citation

The code is released under the MIT licence (`LICENSE`). The labels are derived from GDXray+ and BAM's annotations and are subject to their terms (`DATA_NOTICE.md`). If you use this repository, please cite the paper above and GDXray+:

> D. Mery, V. Riffo, U. Zscherpel, G. Mondragón, I. Lillo, I. Zuccar, H. Lobel, M. Carrasco. GDXray: The database of X-ray images for nondestructive testing. *Journal of Nondestructive Evaluation* 34, 42 (2015).
#   q m l - x r a y - d e f e c t - b e n c h m a r k  
 