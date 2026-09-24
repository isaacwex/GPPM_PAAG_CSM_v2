# GPPM PAAG/CSM — geriatric pneumonia detection: multi-seed ablation

Code, derived data and results for the manuscript *"Attention and Comorbidity-Suppression Modules for Geriatric Pneumonia Detection on Chest Radiographs: A Multi-Seed Ablation Study of DenseNet-121"* (Wekesa et al., submitted to PeerJ).

- Software archive (this repository): DOI `10.5281/zenodo.22945067`
- Trained model weights (12 checkpoints): DOI `10.5281/zenodo.22945068`

## Repository layout
| Path | Contents |
|---|---|
| `data_preparation/PatientLevelDataSplit_LeakProof.ipynb` | Builds the cohort: age ≥ 65, frontal views, label mapping by priority, patient-level 60/20/20 split (random_state = 42) |
| `data_preparation/<renaming script>` | Step that produced the final `CheXpert_<patient>_<age>_<sex>_<view>_…` filenames |
| `data/label_mapping.csv` | Source label → study group mapping and priority order |
| `data/split_assignments.csv` | One row per image: filename, split, group, patient ID, age, sex, view (no images) |
| `data/cohort_description.csv` | Table 1 counts |
| `notebooks/GPPM_PAAG_CSM_v2_multiseed.ipynb` | Training, evaluation, Grad-CAM/IoU, multi-seed runs |
| `analysis/cross_seed_analysis.py` | Per-seed DeLong, random-effects synthesis (Table 8), validation-threshold and Platt-recalibration analyses, IoU Wilcoxon tests |
| `results/` | Metrics (JSON/CSV), per-image test predictions (`predictions.npz`), IoU CSVs and figures for every variant and seed |

## Data access
The images are **not** redistributed. Obtain them from the providers and accept their terms:
- CheXpert v1.0 and VisualCheXbert labels (`train_visualCheXbert.csv`): https://stanfordmlgroup.github.io/competitions/chexpert/
- NIH ChestX-ray8/14 (`Data_Entry_2017_v2020.csv`): https://nihcc.app.box.com/v/ChestXray-NIHCC

## Reproducing the paper
1. `pip install -r requirements.txt` (Python 3.x, PyTorch + torchvision, scikit-learn, scipy, opencv-python, pandas, numpy, matplotlib, seaborn).
2. Edit the paths at the top of `PatientLevelDataSplit_LeakProof.ipynb`, run it, then run the renaming step to create `Training/`, `Validation/`, `Testing/` group folders.
3. In the training notebook set `BASE_DIR`, then for each variant (`base`, `paag`, `csm`, `paag_csm`): mode `train` (seed 42), `train_multiseed` (seeds 101 and 2024), `test`; finally `compare_variants`. Each run took about 6 hours on one GPU.
4. `python analysis/cross_seed_analysis.py results/` reproduces Table 8 and the recalibration/threshold results from the saved predictions, without retraining.

| Manuscript item | Source file |
|---|---|
| Table 1 | `data/cohort_description.csv` |
| Tables 2–3, Figures 5–7 | `results/VariantComparison/` |
| Tables 4–5, Figures 8–12 | `results/TestViz/<variant>/` (test sample), `results/BestModel/<variant>/LocalizationIoU/` (validation replication sample) |
| Table 6 | `results/TestViz/<variant>/*_subgroup_analysis.csv`, `final_test_metrics.json` |
| Tables 7–8 | `results/BestModel/<variant>/seed_*/test_rescored/` + `analysis/cross_seed_analysis.py` |

## Using the trained weights
Download from the weights DOI. Each file is `<variant>_seed<seed>.pth`; load with
`model = build_model_variant(variant, 6, comorbidity_classes, pneumonia_idx, True); model.load_state_dict(torch.load(path))`.
SHA-256 checksums are listed in the weights record.

## Licence
Code: MIT. Model weights and derived tables: CC BY 4.0. Use of the underlying images remains subject to the CheXpert and NIH terms.
