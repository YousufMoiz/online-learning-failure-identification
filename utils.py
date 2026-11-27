"""Shared functions for the online failure identification experiments.

The notebooks import everything from here so that the data loading, the
test stream and the models are defined in one place.
"""

import time
from collections import deque
from copy import deepcopy

import numpy as np
import pandas as pd
from river import drift, forest
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FEATURES = ["Laser current (mA)", "LP power (dBm)", "OSNR (dB)", "BER (dB)"]
CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ["Normal", "ECL", "EDFA", "NLI"]

TRAIN_FILE = "Lightpath_756_label_4_QoT_dataset_train_900_cleaned.csv"
TEST_FILE = "Lightpath_756_label_4_QoT_dataset_test_300_cleaned.csv"

# The test stream has five phases of 5000 samples. Each entry lists the
# classes present in that phase.
PHASE_CLASSES = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 2, 3], [0, 1, 2, 3]]
PHASE_LENGTH = 5000
PHASE_NAMES = ["Normal + ECL", "+ EDFA", "+ NLI", "- ECL", "ECL returns"]


# ---------------------------------------------------------------- data

def load_dataset(data_dir):
    """Read the training and test files of the dataset."""
    train_df = pd.read_csv(f"{data_dir}/{TRAIN_FILE}")
    test_df = pd.read_csv(f"{data_dir}/{TEST_FILE}")
    return train_df, test_df


def make_training_set(train_df, n_per_class=10000, seed=42):
    """Normal and ECL samples used to train every model before the stream.

    Sampled without replacement from the training file, then shuffled.
    """
    parts = [train_df[train_df["Failure type"] == c].sample(n_per_class, random_state=seed)
             for c in (0, 1)]
    df = pd.concat(parts).reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df[FEATURES], df["Failure type"].to_numpy()


def make_oracle_extra(train_df, n_samples=4000, seed=42):
    """EDFA and NLI samples that only the oracle sees before the stream.

    Taken from the training file so that none of them appear in the stream.
    """
    df = train_df[train_df["Failure type"].isin([2, 3])].sample(n_samples, random_state=seed)
    return df[FEATURES], df["Failure type"].to_numpy()


def make_phased_stream(test_df, phase_classes=PHASE_CLASSES,
                       phase_length=PHASE_LENGTH, seed=42):
    """Build the test stream from the test file.

    In every phase, each sample is drawn at random from the classes active in
    that phase, so the classes are mixed rather than arriving in blocks.
    Samples are taken without replacement.

    Returns the features, the labels and the index where each phase starts.
    """
    rng = np.random.default_rng(seed)
    pools = {c: test_df[test_df["Failure type"] == c].sample(frac=1, random_state=seed)
             for c in CLASSES}
    used = {c: 0 for c in CLASSES}

    phases, starts = [], []
    for i, active in enumerate(phase_classes):
        start = i * phase_length
        starts.append(start)
        labels = rng.choice(sorted(active), size=phase_length)
        parts = []
        for c in sorted(active):
            k = int((labels == c).sum())
            parts.append(pools[c].iloc[used[c]:used[c] + k])
            used[c] += k
        phases.append(pd.concat(parts).sample(frac=1, random_state=seed + start))

    stream = pd.concat(phases).reset_index(drop=True)
    return stream[FEATURES], stream["Failure type"].to_numpy(), starts


def fisher_discriminant_ratio(X, y):
    """FDR of each feature (Eq. 4 of the paper) and their mean."""
    y = np.asarray(y)
    ratios = {}
    for f in FEATURES:
        values = X[f].to_numpy()
        overall_mean = values.mean()
        between = within = 0.0
        for c in np.unique(y):
            v = values[y == c]
            between += len(v) * (v.mean() - overall_mean) ** 2
            within += len(v) * v.var(ddof=1)
        ratios[f] = between / within
    return ratios, float(np.mean(list(ratios.values())))


# -------------------------------------------------------------- models

def make_arf(seed=42, drift_detection=True):
    """Adaptive Random Forest with River's default settings.

    With drift_detection=False the per-tree warning and drift detectors are
    switched off. Passing None does not do this in River (it falls back to
    ADWIN), so NoDrift is used instead.
    """
    if drift_detection:
        return forest.ARFClassifier(n_models=10, max_features=2, seed=seed)
    return forest.ARFClassifier(n_models=10, max_features=2, seed=seed,
                                drift_detector=drift.NoDrift(),
                                warning_detector=drift.NoDrift())


def make_batch_forest(seed=42):
    """Batch random forest used by the periodic and oracle models."""
    return RandomForestClassifier(n_estimators=10, max_features=2,
                                  random_state=seed, n_jobs=-1)


class OnlineMLP:
    """Small scikit-learn MLP updated with one sample at a time.

    It has the same predict_one / predict_proba_one / learn_one methods as a
    River model, so it can be passed to run_stream. Features are standardised
    with the statistics of the initial training set.
    """

    def __init__(self, hidden_layers=(32, 16), seed=42):
        self.scaler = StandardScaler()
        self.net = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=seed)

    def fit_initial(self, X, y, epochs=5, batch_size=256):
        Xs = self.scaler.fit_transform(X.to_numpy())
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            order = rng.permutation(len(y))
            for i in range(0, len(y), batch_size):
                batch = order[i:i + batch_size]
                self.net.partial_fit(Xs[batch], y[batch], classes=CLASSES)
        return self

    def _row(self, x):
        return self.scaler.transform([[x[f] for f in FEATURES]])

    def predict_one(self, x):
        return int(self.net.predict(self._row(x))[0])

    def predict_proba_one(self, x):
        return dict(zip(self.net.classes_, self.net.predict_proba(self._row(x))[0]))

    def learn_one(self, x, label):
        self.net.partial_fit(self._row(x), [label])


def train_river_model(model, X, y):
    """Train a River model one sample at a time."""
    for x, label in zip(X.to_dict("records"), y):
        model.learn_one(x, int(label))
    return model


def class_probabilities(model, x):
    """River returns a dict of probabilities; turn it into a fixed-length row."""
    proba = model.predict_proba_one(x)
    return [proba.get(c, 0.0) for c in CLASSES]


def run_stream(model, X, y, learn=True, label_delay=0, label_fraction=1.0,
               label_noise=0.0, seed=0):
    """Test-then-train evaluation of a River model on the stream.

    Every sample is first predicted. If learn is True, its label is then given
    to the model, possibly late (label_delay samples later), only for a random
    share of the samples (label_fraction) and possibly wrong (label_noise is
    the share of labels replaced by a wrong class). A wrong label is only
    drawn from the failure types that have already appeared in the stream,
    since nobody can mislabel a sample as a failure they have never seen.

    The model passed in is copied, so the same trained model can be reused.
    """
    model = deepcopy(model)
    rng = np.random.default_rng(seed)
    waiting = deque()
    seen_classes = set()
    preds, probas = [], []

    for t, (x, label) in enumerate(zip(X.to_dict("records"), y)):
        preds.append(model.predict_one(x))
        probas.append(class_probabilities(model, x))
        if not learn:
            continue

        label = int(label)
        seen_classes.add(label)
        if rng.random() < label_fraction:
            if rng.random() < label_noise:
                wrong = sorted(seen_classes - {label})
                if wrong:
                    label = int(rng.choice(wrong))
            waiting.append((t + label_delay, x, label))

        while waiting and waiting[0][0] <= t:
            _, x_old, label_old = waiting.popleft()
            model.learn_one(x_old, label_old)

    return np.array(preds), np.array(probas)


def run_batch_retraining(X_train, y_train, X_stream, y_stream, retrain_at, seed=42):
    """Batch forest rebuilt from scratch at the given stream positions.

    Each rebuild uses the initial training set plus every stream sample seen so
    far. Between rebuilds the model only predicts. Also returns the time taken
    by each rebuild in seconds.
    """
    X_stream = X_stream.to_numpy()
    model = make_batch_forest(seed).fit(X_train.to_numpy(), y_train)
    points = sorted(t for t in set(retrain_at) if 0 < t < len(y_stream)) + [len(y_stream)]

    preds = np.empty(len(y_stream), dtype=int)
    probas = np.zeros((len(y_stream), len(CLASSES)))
    fit_times = []
    start = 0
    for t in points:
        preds[start:t] = model.predict(X_stream[start:t])
        probas[start:t, model.classes_] = model.predict_proba(X_stream[start:t])
        if t < len(y_stream):
            X_all = np.vstack([X_train.to_numpy(), X_stream[:t]])
            y_all = np.concatenate([y_train, y_stream[:t]])
            tic = time.perf_counter()
            model = make_batch_forest(seed).fit(X_all, y_all)
            fit_times.append(time.perf_counter() - tic)
        start = t
    return preds, probas, fit_times


def run_batch_model(model, X_stream):
    """Predictions of an already trained batch model on the whole stream."""
    probas = np.zeros((len(X_stream), len(CLASSES)))
    probas[:, model.classes_] = model.predict_proba(X_stream.to_numpy())
    return model.predict(X_stream.to_numpy()), probas


# ------------------------------------------------------------- metrics

def rolling_accuracy(y, preds, window=500, step=40):
    """Accuracy over a sliding window, evaluated every `step` samples."""
    correct = (np.asarray(preds) == np.asarray(y)).astype(float)
    ends = np.arange(window, len(y) + 1, step)
    return ends, np.array([correct[e - window:e].mean() for e in ends])


def window_auc(y, probas):
    """One-vs-rest AUC averaged over the classes present in the window.

    Classes that are absent from the window are skipped because their AUC is
    undefined. Probabilities are used as given by the model.
    """
    y = np.asarray(y)
    scores = []
    for c in CLASSES:
        is_c = y == c
        if 0 < is_c.sum() < len(y):
            scores.append(roc_auc_score(is_c, probas[:, c]))
    return np.mean(scores) if scores else np.nan


def rolling_auc(y, probas, window=500, step=40):
    ends = np.arange(window, len(y) + 1, step)
    return ends, np.array([window_auc(y[e - window:e], probas[e - window:e]) for e in ends])


def per_class_recall(y, preds):
    y, preds = np.asarray(y), np.asarray(preds)
    return {CLASS_NAMES[c]: (preds[y == c] == c).mean() for c in CLASSES if (y == c).any()}


def recovery_time(y, preds, onset, window=200, max_error=0.01, limit=PHASE_LENGTH):
    """Samples after `onset` until the error rate over the next `window`
    samples falls below `max_error`. Returns NaN if that never happens
    within `limit` samples.
    """
    wrong = (np.asarray(preds) != np.asarray(y)).astype(float)
    for k in range(limit - window + 1):
        if wrong[onset + k:onset + k + window].mean() < max_error:
            return k
    return np.nan


# ------------------------------------------------------- drift detection

PH_SETTINGS = dict(min_instances=100, delta=0.005, threshold=50, alpha=0.9999, mode="both")


def pht_ks_alarms(values, ks_window=500, ks_alpha=0.005):
    """Two-stage drift detection on one telemetry feature.

    Page-Hinkley gives candidate alarms. A candidate is confirmed if a
    Kolmogorov-Smirnov test finds that the two windows of ks_window samples
    before it come from different distributions.
    """
    values = np.asarray(values, dtype=float)
    ph = drift.PageHinkley(**PH_SETTINGS)
    candidates, confirmed = [], []
    for t, v in enumerate(values):
        ph.update(v)
        if not ph.drift_detected:
            continue
        candidates.append(t)
        older = values[max(0, t - 2 * ks_window):t - ks_window]
        recent = values[max(0, t - ks_window):t]
        if len(older) >= 30 and len(recent) >= 30 and ks_2samp(older, recent).pvalue < ks_alpha:
            confirmed.append(t)
    return candidates, confirmed


def merge_close_alarms(alarms, min_gap=800):
    """Keep only the first alarm of every group closer than min_gap samples."""
    merged = []
    for t in sorted(alarms):
        if not merged or t - merged[-1] > min_gap:
            merged.append(t)
    return merged
