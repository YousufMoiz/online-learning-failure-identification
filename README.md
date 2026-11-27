# Online learning for soft-failure identification under concept drift

Code for the paper *"Online learning-based concept drift adaptation for failure identification in optical networks"* (submitted to the Journal of Optical Communications and Networking).

The experiments train an Adaptive Random Forest (ARF) on normal and ECL-failure telemetry, then run it on a test stream in which EDFA and NLI failures appear, ECL disappears and later returns. The online model, which keeps learning from the stream, is compared with a static model, a periodically retrained model and an oracle that sees every failure type in advance.

## Data

The experiments use the optical network soft failure dataset of Ghosh and Adhya:
https://data.mendeley.com/datasets/y3pspy7j83/1

Download it and put these two files in a folder called `data` next to the notebooks:

```
data/Lightpath_756_label_4_QoT_dataset_train_900_cleaned.csv
data/Lightpath_756_label_4_QoT_dataset_test_300_cleaned.csv
```

The training set (10,000 normal and 10,000 ECL samples) is drawn from the first file, and the test stream from the second, so no sample appears in both.

## Setup

The results in the paper were produced with Python 3.10.

```
pip install -r requirements.txt
```

## Notebooks

Run `01` first if you want to check the data; the other notebooks do not depend on each other. Each one rebuilds the training set and the stream from the raw files and saves its figures in `figures/` and its tables as CSV files.

| Notebook | Contents | Paper | Run time* |
|---|---|---|---|
| `01_data_preparation` | Training set, test stream, overlap check, feature distributions, Fisher discriminant ratio, pairwise plot, correlation matrix | Fig. 3, Appendix A (Figs. 9, 10) | ~2 min |
| `02_main_results` | Static, online, periodic and oracle models: rolling accuracy and AUC, misclassifications, confusion matrices, per-class recall and macro-F1, ten seeds, other class orders, latency | Figs. 5–8, Section 4 | ~30 min |
| `03_label_availability` | Online model with delayed labels, a label budget and label noise | Section 4 | ~30 min |
| `04_baselines_ablation` | Retraining schedules, other online learners, decision trees, ARF with and without its drift detectors, hyperparameter sensitivity | Section 4 | ~1.5 h |
| `05_drift_detection` | Page-Hinkley + Kolmogorov-Smirnov drift detection on each feature | Fig. 4, Section 3 | ~2 min |

\*Rough times on a laptop. Lower `N_SEEDS` (and `GRID_SEEDS` in `04`) at the top of a notebook for a quicker run.

`utils.py` holds everything the notebooks share: data loading, the stream construction, the models and the metrics.

## Reproducibility

All randomness is seeded. Runs on the same machine give identical results, but River's models can give slightly different numbers on a different operating system or Python version (for example, 113 versus 122 misclassifications for the online model with seed 42). The averages over ten seeds should agree within their standard deviation.

## Citation

If you use this code, please cite the paper (reference to be added on publication).
