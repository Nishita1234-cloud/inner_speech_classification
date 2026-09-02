import os
from pathlib import Path
import numpy as np
import mne
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.pipeline import Pipeline
from mne.viz import plot_topomap
import matplotlib.pyplot as plt

DERIV_ROOT = Path("/kaggle/input/datasets/nishp101/preprocessed-nemar-inner-speech/ds003626/derivatives")
TARGET_SFREQ = 256.0
CROP_TMIN, CROP_TMAX = 0.5, 3.0
CLASS_MAP = {"Arriba": 0, "Abajo": 1, "Derecha": 2, "Izquierda": 3}
CLASS_NAMES = ["Arriba", "Abajo", "Derecha", "Izquierda"]
FREQ_BANDS = {"theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
N_SEEDS = 20
WITHIN_SUBJECT_SEEDS = list(range(N_SEEDS))
np.random.seed(0)

CORRUPTED_SESSIONS = {("sub-09", "ses-01"), ("sub-10", "ses-03")}

# Load + preprocess
def load_subject_data(sub_dir):
    all_epochs = []
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        fif_files = list(ses_dir.glob("*_eeg-epo.fif"))
        if not fif_files:
            continue
        fpath = fif_files[0]
        try:
            epochs = mne.read_epochs(str(fpath), preload=True, verbose="ERROR")
        except Exception as e:
            print(f"    SKIP {fpath.name}: {type(e).__name__}: {e}")
            continue
        events_files = list(ses_dir.glob("*_events.dat"))
        events_arr = pd.read_pickle(events_files[0])
        event_ids = epochs.events[:, 2]
        id_to_name = {v: k for k, v in epochs.event_id.items()}
        y_check = np.array([CLASS_MAP[id_to_name[eid]] for eid in event_ids])
        inner_speech_mask = events_arr[:, 2] == 1
        epochs = epochs[inner_speech_mask]
        print(f"    {ses_dir.name}: kept {inner_speech_mask.sum()}/{len(inner_speech_mask)} trials (inner speech only)")
        all_epochs.append(epochs)
    if not all_epochs:
        return None, None
    epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]
    epochs = epochs.pick_types(eeg=True)
    epochs = epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    if epochs.info["sfreq"] > TARGET_SFREQ * 1.05:
        epochs = epochs.resample(TARGET_SFREQ)
    X = epochs.get_data() * 1e6
    event_ids = epochs.events[:, 2]
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    y = np.array([CLASS_MAP[id_to_name[eid]] for eid in event_ids])
    return X, y, epochs


def split_80_10_10(X, y, seed):
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_by_subject, y_by_subject = {}, {}
REFERENCE_EPOCHS = None
for sub_dir in sorted(DERIV_ROOT.glob("sub-*")):
    print(f"  {sub_dir.name}")
    result = load_subject_data(sub_dir)
    if result[0] is None:
        continue
    X, y, epochs = result
    X_by_subject[sub_dir.name] = X
    y_by_subject[sub_dir.name] = y
    if REFERENCE_EPOCHS is None:
        REFERENCE_EPOCHS = epochs
print(f"Loaded {len(X_by_subject)} subjects")

ORIGINAL_CH_NAMES = REFERENCE_EPOCHS.ch_names
montage = mne.channels.make_standard_montage("biosemi128")
pos_dict = montage.get_positions()["ch_pos"]
name_to_idx = {ch: i for i, ch in enumerate(ORIGINAL_CH_NAMES)}

standard_pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
standard_names = list(standard_pos.keys())
standard_coords = np.array([standard_pos[ch] for ch in standard_names])
def _nearest_1005_name(ch):
    dists = np.linalg.norm(standard_coords - np.array(pos_dict[ch]), axis=1)
    return standard_names[np.argmin(dists)]

#got from mapping
ROI_CHANNELS = {
    "IF": ['B25', 'B26', 'B27', 'C7', 'C8', 'C30', 'D7', 'D8', 'D22', 'D23'],
    "PM": ['A1', 'B20', 'B21', 'B22', 'B23', 'B24', 'B28', 'B29', 'B30', 'B31', 'B32',
           'C1', 'C2', 'C21', 'C23', 'D1', 'D2', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14',
           'D18', 'D19', 'D20', 'D21'],
    "TMP": ['B10', 'B11', 'B12', 'B14', 'B15', 'D24', 'D25', 'D30', 'D31', 'D32'],
    "PAR": ['A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14',
            'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A26', 'A27', 'A28', 'A29',
            'A30', 'A31', 'A32', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B13', 'D29'],
}

_missing = [ch for chans in ROI_CHANNELS.values() for ch in chans if ch not in name_to_idx]
if _missing:
    raise ValueError(f"These hardcoded ROI channels aren't in the loaded data: {_missing}")

ROI_IDX = {label: [name_to_idx[ch] for ch in chans] for label, chans in ROI_CHANNELS.items()}
ROI_LEFT_IDX = {label: [name_to_idx[ch] for ch in chans if pos_dict[ch][0] < 0] for label, chans in ROI_CHANNELS.items()}
ROI_RIGHT_IDX = {label: [name_to_idx[ch] for ch in chans if pos_dict[ch][0] > 0] for label, chans in ROI_CHANNELS.items()}

for label, chans in ROI_CHANNELS.items():
    print(f"{label}: {len(chans)} channels ({len(ROI_LEFT_IDX[label])} left / {len(ROI_RIGHT_IDX[label])} right)")
    print("  " + ", ".join(f"{ch}({_nearest_1005_name(ch)})" for ch in chans))

# Sanity-check plot to make sure ROI selection actually landed in right spot
SAMPLE_INFO = REFERENCE_EPOCHS.copy().info
SAMPLE_INFO.set_montage("biosemi128", on_missing="warn")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
mask_a = np.zeros(len(ORIGINAL_CH_NAMES))
for ch in ROI_CHANNELS["IF"]: mask_a[name_to_idx[ch]] = 1.0
for ch in ROI_CHANNELS["TMP"]: mask_a[name_to_idx[ch]] = -1.0
plot_topomap(mask_a, SAMPLE_INFO, axes=axes[0], show=False, cmap="PiYG", vlim=(-1, 1), sensors=True, contours=0)
axes[0].set_title("IF (green) vs TMP (pink)")

mask_b = np.zeros(len(ORIGINAL_CH_NAMES))
for ch in ROI_CHANNELS["PM"]: mask_b[name_to_idx[ch]] = 1.0
for ch in ROI_CHANNELS["PAR"]: mask_b[name_to_idx[ch]] = -1.0
plot_topomap(mask_b, SAMPLE_INFO, axes=axes[1], show=False, cmap="PuOr", vlim=(-1, 1), sensors=True, contours=0)
axes[1].set_title("PM (orange) vs PAR (purple)")
plt.tight_layout()
plt.savefig("roi_sanity_check.png", dpi=150)
plt.show()
print("Check roi_sanity_check.png -- IF should sit frontal/fronto-temporal, TMP posterior-temporal,")
print("PM central/frontocentral, PAR should span central-parietal through occipito-parietal.")

EPS = 1e-12
INCLUDE_HJORTH = True
INCLUDE_CONNECTIVITY = True
LOG_TRANSFORM_ABS_POWER = True

def compute_band_powers(X, sfreq, freq_bands):
    nperseg = min(256, X.shape[-1])
    freqs, psd = welch(X, fs=sfreq, nperseg=nperseg, axis=-1)
    out = {}
    for band, (lo, hi) in freq_bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        out[band] = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return out

def region_mean(power_arr, idx_list):
    return power_arr[:, idx_list].mean(axis=1)

def compute_hjorth(X):
    activity = X.var(axis=-1)
    dx = np.diff(X, axis=-1)
    mobility = np.sqrt(dx.var(axis=-1) / (activity + EPS))
    ddx = np.diff(dx, axis=-1)
    mobility_dx = np.sqrt(ddx.var(axis=-1) / (dx.var(axis=-1) + EPS))
    complexity = mobility_dx / (mobility + EPS)
    return activity, mobility, complexity

def compute_connectivity(X, idx_a, idx_b):
    sig_a = X[:, idx_a, :].mean(axis=1)
    sig_b = X[:, idx_b, :].mean(axis=1)
    sig_a = sig_a - sig_a.mean(axis=-1, keepdims=True)
    sig_b = sig_b - sig_b.mean(axis=-1, keepdims=True)
    num = (sig_a * sig_b).sum(axis=-1)
    den = np.sqrt((sig_a ** 2).sum(axis=-1) * (sig_b ** 2).sum(axis=-1)) + EPS
    return num / den

def extract_roi_features(X, sfreq):
    bands = dict(FREQ_BANDS)
    bp = compute_band_powers(X, sfreq, bands)

    features = []

    def add_power_features(label, idx_list, left_idx, right_idx):
        theta = region_mean(bp["theta"], idx_list)
        alpha = region_mean(bp["alpha"], idx_list)
        beta = region_mean(bp["beta"], idx_list)
        abs_vals = {"theta": theta, "alpha": alpha, "beta": beta}
        total = theta + alpha + beta + EPS
        for band, val in abs_vals.items():
            out_val = np.log(val + EPS) if LOG_TRANSFORM_ABS_POWER else val
            suffix = " (log)" if LOG_TRANSFORM_ABS_POWER else ""
            features.append((f"{label} {band} power{suffix}", out_val, idx_list))
        for band, val in abs_vals.items():
            features.append((f"{label} {band} rel power", val / total, idx_list))
        alpha_L, alpha_R = region_mean(bp["alpha"], left_idx), region_mean(bp["alpha"], right_idx)
        beta_L, beta_R = region_mean(bp["beta"], left_idx), region_mean(bp["beta"], right_idx)
        features.append((f"{label} alpha lateralization", (alpha_R - alpha_L) / (alpha_R + alpha_L + EPS), idx_list))
        features.append((f"{label} beta lateralization", (beta_R - beta_L) / (beta_R + beta_L + EPS), idx_list))

    for label in ROI_CHANNELS:
        add_power_features(label, ROI_IDX[label], ROI_LEFT_IDX[label], ROI_RIGHT_IDX[label])

    if INCLUDE_HJORTH:
        activity, mobility, complexity = compute_hjorth(X)
        for label in ROI_CHANNELS:
            idx_list = ROI_IDX[label]
            features.append((f"{label} Hjorth activity", region_mean(activity, idx_list), idx_list))
            features.append((f"{label} Hjorth mobility", region_mean(mobility, idx_list), idx_list))
            features.append((f"{label} Hjorth complexity", region_mean(complexity, idx_list), idx_list))

    if INCLUDE_CONNECTIVITY:
        conn = compute_connectivity(X, ROI_IDX["IF"], ROI_IDX["TMP"])
        features.append(("IF-TMP connectivity (Pearson r)", conn, ROI_IDX["IF"] + ROI_IDX["TMP"]))

    X_feat = np.column_stack([vals for _, vals, _ in features])
    feature_names = [name for name, _, _ in features]
    feature_region_idx = [region for _, _, region in features]
    return X_feat, feature_names, feature_region_idx

# Within-subject loop: 20 seeds per subject, Linear SVM
FEATURE_NAMES, FEATURE_TO_REGION_IDX = None, None
subject_test_acc, subject_macro_f1 = {}, {}
subject_per_class_f1 = {}
cm_sum_all = np.zeros((4, 4), dtype=int)
coef_importance_all = []
perm_importance_all = []

for sub_id in sorted(X_by_subject.keys()):
    X_sub = X_by_subject[sub_id]
    y_sub = y_by_subject[sub_id]
    X_feat, feat_names, feat_region_idx = extract_roi_features(X_sub, TARGET_SFREQ)
    if FEATURE_NAMES is None:
        FEATURE_NAMES, FEATURE_TO_REGION_IDX = feat_names, feat_region_idx
    print(f"\n=== {sub_id} ({X_sub.shape[0]} trials, {N_SEEDS} seeds, {X_feat.shape[1]} features) ===")

    seed_acc, seed_f1_macro, seed_f1_per_class = [], [], []
    cm_sum_sub = np.zeros((4, 4), dtype=int)

    for seed in WITHIN_SUBJECT_SEEDS:
        X_tr, y_tr, X_val, y_val, X_te, y_te = split_80_10_10(X_feat, y_sub, seed)

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_te_s = scaler.transform(X_te)

        clf = LinearSVC(class_weight="balanced", random_state=seed, max_iter=10000) #tackles class imabalance too
        clf.fit(X_tr_s, y_tr)

        y_pred = clf.predict(X_te_s)
        acc = balanced_accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average="macro")
        f1_per_class = f1_score(y_te, y_pred, average=None, labels=[0, 1, 2, 3])
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2, 3])

        seed_acc.append(acc)
        seed_f1_macro.append(f1_macro)
        seed_f1_per_class.append(f1_per_class)
        cm_sum_sub += cm
        coef_importance_all.append(np.abs(clf.coef_).mean(axis=0))

        # Permutation importance is only computed for seed 0's model, not all 20 for time purposes
        if seed == 0:
            perm_result = permutation_importance(
                clf, X_val_s, y_val, n_repeats=30, random_state=0, scoring="balanced_accuracy"
            )
            perm_importance_all.append(perm_result.importances_mean)

    seed_acc = np.array(seed_acc)
    seed_f1_macro = np.array(seed_f1_macro)
    seed_f1_per_class = np.array(seed_f1_per_class)  # (n_seeds, 4)

    subject_test_acc[sub_id] = seed_acc
    subject_macro_f1[sub_id] = seed_f1_macro
    subject_per_class_f1[sub_id] = seed_f1_per_class
    cm_sum_all += cm_sum_sub

    print(f"  SVM balanced accuracy across {N_SEEDS} seeds: {seed_acc.mean():.3f} +/- {seed_acc.std():.3f} "
          f"(min={seed_acc.min():.3f}, max={seed_acc.max():.3f})")
    print(f"  SVM macro-F1 across {N_SEEDS} seeds: {seed_f1_macro.mean():.3f} +/- {seed_f1_macro.std():.3f}")
    per_class_str = ", ".join(f"{CLASS_NAMES[c]}={seed_f1_per_class[:, c].mean():.3f}" for c in range(4))
    print(f"  Per-class F1 (mean across seeds): {per_class_str}")

# Cross-subject aggregation + summary
subject_means = np.array([subject_test_acc[s].mean() for s in sorted(subject_test_acc)])
subject_ids_sorted = sorted(subject_test_acc)
all_accs_pooled = np.concatenate([subject_test_acc[s] for s in subject_ids_sorted])

print(f"\nSVM within-subject test balanced accuracy (mean of per-subject means): "
      f"{subject_means.mean():.3f} +/- {subject_means.std():.3f}")
print(f"SVM within-subject test balanced accuracy (pooled across all {len(all_accs_pooled)} subject-seed runs): "
      f"{all_accs_pooled.mean():.3f} +/- {all_accs_pooled.std():.3f}")
for s in subject_ids_sorted:
    print(f"  {s}: {subject_test_acc[s].mean():.3f} +/- {subject_test_acc[s].std():.3f} (across {N_SEEDS} seeds)")

macro_f1_means = np.array([subject_macro_f1[s].mean() for s in subject_ids_sorted])
print(f"\nSVM within-subject macro-F1 (mean of per-subject means): "
      f"{macro_f1_means.mean():.3f} +/- {macro_f1_means.std():.3f}")

per_class_f1_all = np.concatenate([subject_per_class_f1[s] for s in subject_ids_sorted], axis=0)  # (total_seeds, 4)
print("Per-class F1 (mean +/- std, pooled across all subject-seed runs):")
for c in range(4):
    print(f"  {CLASS_NAMES[c]}: {per_class_f1_all[:, c].mean():.3f} +/- {per_class_f1_all[:, c].std():.3f}")

# Plot 1: per-subject accuracy box plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([subject_test_acc[s] for s in subject_ids_sorted], labels=subject_ids_sorted)
ax.axhline(0.25, color="gray", linestyle="--", label="chance (0.25)")
ax.set_ylabel("Balanced accuracy")
ax.set_title(f"SVM within-subject test accuracy across {N_SEEDS} seeds, all subjects")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("svm_per_subject_accuracy_boxplot.png", dpi=150)
plt.show()

# Plot 2: pooled confusion matrix
cm_norm = cm_sum_all.astype(float) / cm_sum_all.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES, rotation=45)
ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"SVM pooled confusion matrix\n(summed across all subjects and {N_SEEDS} seeds, row-normalized)")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                 color="white" if cm_norm[i, j] > 0.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("svm_pooled_confusion_matrix.png", dpi=150)
plt.show()

# Plot 3: per-class F1 summary
fig, ax = plt.subplots(figsize=(6, 4))
means = [per_class_f1_all[:, c].mean() for c in range(4)]
stds = [per_class_f1_all[:, c].std() for c in range(4)]
ax.bar(CLASS_NAMES, means, yerr=stds, capsize=4, color="#4C72B0")
ax.set_ylabel("F1 score")
ax.set_title("SVM per-class F1 (mean +/- std, pooled across subjects and seeds)")
plt.tight_layout()
plt.savefig("svm_per_class_f1.png", dpi=150)
plt.show()

#Plot 4: SVM coefficient importance
coef_importance_all = np.array(coef_importance_all)
coef_mean = coef_importance_all.mean(axis=0)
coef_std = coef_importance_all.std(axis=0)
order = np.argsort(coef_mean)[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(np.array(FEATURE_NAMES)[order], coef_mean[order], xerr=coef_std[order], color="#55A868")
ax.set_xlabel("Mean |SVM coefficient| (averaged across one-vs-rest classifiers)")
ax.set_title("SVM feature importance by coefficient magnitude\n(mean +/- std across all subjects and seeds)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("svm_coefficient_importance.png", dpi=150)
plt.show()

# Plot 5: permutation importance
perm_importance_all = np.array(perm_importance_all)
perm_mean = perm_importance_all.mean(axis=0)
perm_std = perm_importance_all.std(axis=0)
order_perm = np.argsort(perm_mean)[::-1]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(np.array(FEATURE_NAMES)[order_perm], perm_mean[order_perm], xerr=perm_std[order_perm], color="#C44E52")
ax.set_xlabel("Mean permutation importance (drop in balanced accuracy, validation set)")
ax.set_title("SVM feature importance by permutation\n(mean +/- std across subjects, representative seed=0 model)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("svm_permutation_importance.png", dpi=150)
plt.show()

#Plot 6: aggregated importance mapped back onto the scalp
def feature_importance_to_topomap(importance_per_feature):
    chan_importance = np.zeros(len(ORIGINAL_CH_NAMES))
    for feat_i, imp in enumerate(importance_per_feature):
        idx_list = FEATURE_TO_REGION_IDX[feat_i]
        for orig_i in idx_list:
            chan_importance[orig_i] += imp / len(idx_list)
    return chan_importance

def normalize01(v):
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + EPS)

combined_importance = 0.5 * normalize01(coef_mean) + 0.5 * normalize01(perm_mean)
chan_importance = feature_importance_to_topomap(combined_importance)

fig, ax = plt.subplots(figsize=(5, 5))
im, _ = plot_topomap(chan_importance, SAMPLE_INFO, axes=ax, show=False, cmap="Blues",
                      sensors=True, contours=4)
ax.set_title("SVM feature importance mapped to scalp\n(combined coefficient + permutation importance,\nspread across each feature's ROI channels)")
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig("svm_importance_topomap.png", dpi=150)
plt.show()
