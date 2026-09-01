import os
from pathlib import Path
import numpy as np
import mne
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from braindecode.models import EEGNet
from scipy import stats
import matplotlib.pyplot as plt
from mne.viz import plot_topomap
import pandas as pd

DERIV_ROOT = Path("path to data in kaggle")
TARGET_SFREQ = 256.0
CROP_TMIN, CROP_TMAX = 0.5, 3.0
CLASS_MAP = {"Arriba": 0, "Abajo": 1, "Derecha": 2, "Izquierda": 3}
CLASS_NAMES = ["Arriba", "Abajo", "Derecha", "Izquierda"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Device:", DEVICE)

#Load and preprocess

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
        if not events_files:
            print(f"    WARNING: no events.dat for {fpath.name}, skipping session (can't verify condition)")
            continue
        events_arr = pd.read_pickle(events_files[0])

        event_ids = epochs.events[:, 2]
        id_to_name = {v: k for k, v in epochs.event_id.items()}
        y_check = np.array([CLASS_MAP[id_to_name[eid]] for eid in event_ids])

        inner_speech_mask = events_arr[:, 2] == 1   # 0=Pronounced, 1=Inner speech, 2=Visualized
        epochs = epochs[inner_speech_mask]

        print(f"    {ses_dir.name}: kept {inner_speech_mask.sum()}/{len(inner_speech_mask)} trials (inner speech only)")
        all_epochs.append(epochs)

    if not all_epochs:
        return None
    epochs = mne.concatenate_epochs(all_epochs) if len(all_epochs) > 1 else all_epochs[0]
    epochs = epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    if epochs.info["sfreq"] > TARGET_SFREQ * 1.05:
        epochs = epochs.resample(TARGET_SFREQ)
    X = epochs.get_data() * 1e6
    event_ids = epochs.events[:, 2]
    id_to_name = {v: k for k, v in epochs.event_id.items()}
    y = np.array([CLASS_MAP[id_to_name[eid]] for eid in event_ids])
    return X, y, epochs.info
    
X_by_subject = {}
y_by_subject = {}
SAMPLE_INFO = None
for sub_dir in sorted(DERIV_ROOT.glob("sub-*")):
    print(f"\nLoading {sub_dir.name}...")
    result = load_subject_data(sub_dir)
    if result is None:
        print(f"  {sub_dir.name}: no usable data, skipping")
        continue
    X, y, info = result
    X_by_subject[sub_dir.name] = X
    y_by_subject[sub_dir.name] = y
    if SAMPLE_INFO is None:
        SAMPLE_INFO = info
    print(f"  {sub_dir.name}: {X.shape[0]} trials total")
    
print(f"\nLoaded {len(X_by_subject)} subjects")


#Per subject zscore normalization
def normalize_per_subject(X):
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std
for sub_id in X_by_subject:
    X_by_subject[sub_id] = normalize_per_subject(X_by_subject[sub_id])
    print(f"  {sub_id}: normalized (per-channel mean~0, std~1)")

N_TIMES = next(iter(X_by_subject.values())).shape[-1]   # same for every subject (same crop + resample), used by EEGNet's n_times
print(f"\nTimesteps per trial: {N_TIMES}")

def eegnet():
    return EEGNet(
        n_chans=128, n_outputs=4, n_times=N_TIMES,
        kernel_length=87, F1=8, D=2, F2=8,
    )

#training / eval helpers 
def predict_in_batches(model, X_t, batch_size=64):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            batch_preds = model(X_t[i:i + batch_size]).argmax(1).cpu().numpy()
            preds.append(batch_preds)
    return np.concatenate(preds)

def evaluate_in_batches(model, X_t, y, batch_size=64):
    return balanced_accuracy_score(y, predict_in_batches(model, X_t, batch_size))

def train_one_fold(model_fn, X_tr, y_tr, X_val, y_val, epochs=60, batch_size=32, lr=1e-3, needs_channel_dim=True):
    model = model_fn().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    X_tr_np = torch.tensor(X_tr, dtype=torch.float32)
    X_val_np = torch.tensor(X_val, dtype=torch.float32)
    if needs_channel_dim:
        X_tr_np = X_tr_np.unsqueeze(1)
        X_val_np = X_val_np.unsqueeze(1)
    X_tr_t = X_tr_np.to(DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(DEVICE)
    X_val_t = X_val_np.to(DEVICE)
    train_acc_history, val_acc_history = [], []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_tr_t.shape[0])
        for i in range(0, X_tr_t.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
        train_acc_history.append(evaluate_in_batches(model, X_tr_t, y_tr))
        val_acc_history.append(evaluate_in_batches(model, X_val_t, y_val))
    final_val_preds = predict_in_batches(model, X_val_t)
    return model, train_acc_history, val_acc_history, final_val_preds

# Within-subject evaluation: repeated random 80/10/10 within subject
N_SEEDS = 20
WITHIN_SUBJECT_SEEDS = list(range(N_SEEDS))
WITHIN_SUBJECT_EPOCHS = 60

os.makedirs("eegnet_within_subject_weights", exist_ok=True)

def split_80_10_10(X, y, seed):
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=seed
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

within_subject_results = {}   

for sub_id in sorted(X_by_subject.keys()):
    X_sub, y_sub = X_by_subject[sub_id], y_by_subject[sub_id]
    n_total = X_sub.shape[0]
    print(f"\n=== Within-subject {sub_id} ({n_total} trials, {N_SEEDS} seeds) ===")

    eegnet_test_accs, eegnet_train_accs, eegnet_val_accs = [], [], []
    test_cms = []
    representative_model = None   

    for seed in WITHIN_SUBJECT_SEEDS:
        X_tr, y_tr, X_val, y_val, X_te, y_te = split_80_10_10(X_sub, y_sub, seed=seed)

        model, train_hist, val_hist, val_preds = train_one_fold(
            eegnet, X_tr, y_tr, X_val, y_val, epochs=WITHIN_SUBJECT_EPOCHS, needs_channel_dim=False
        )

        X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
        test_preds = predict_in_batches(model, X_te_t)
        sub_test_acc = balanced_accuracy_score(y_te, test_preds)
        test_cm = confusion_matrix(y_te, test_preds, labels=list(range(4)))

        eegnet_test_accs.append(sub_test_acc)
        eegnet_train_accs.append(train_hist[-1])
        eegnet_val_accs.append(val_hist[-1])
        test_cms.append(test_cm)

        if seed == 0:
            representative_model = model
            torch.save(model.state_dict(), f"eegnet_within_subject_weights/{sub_id}_seed0.pt")

    eegnet_test_accs = np.array(eegnet_test_accs)
    within_subject_results[sub_id] = {
        "model": representative_model,
        "eegnet_test_accs": eegnet_test_accs,
        "eegnet_train_accs": np.array(eegnet_train_accs),
        "eegnet_val_accs": np.array(eegnet_val_accs),
        "test_cm_sum": np.sum(test_cms, axis=0),
    }
    print(f"  EEGNet test acc across {N_SEEDS} seeds: {eegnet_test_accs.mean():.3f} +/- {eegnet_test_accs.std():.3f}  "
          f"(min={eegnet_test_accs.min():.3f}, max={eegnet_test_accs.max():.3f})")

#Aggregate across subjects
subject_ids = sorted(within_subject_results.keys())
eegnet_subject_means = np.array([within_subject_results[s]["eegnet_test_accs"].mean() for s in subject_ids])
eegnet_subject_stds  = np.array([within_subject_results[s]["eegnet_test_accs"].std() for s in subject_ids])
eegnet_all_accs = np.concatenate([within_subject_results[s]["eegnet_test_accs"] for s in subject_ids])   # 200 values

print(f"\nEEGNet within-subject test balanced accuracy (mean of per-subject means): "
      f"{eegnet_subject_means.mean():.3f} +/- {eegnet_subject_means.std():.3f}")
print(f"EEGNet within-subject test balanced accuracy (pooled across all {len(eegnet_all_accs)} subject-seed runs): "
      f"{eegnet_all_accs.mean():.3f} +/- {eegnet_all_accs.std():.3f}")
for s, m, sd in zip(subject_ids, eegnet_subject_means, eegnet_subject_stds):
    print(f"  {s}: {m:.3f} +/- {sd:.3f}  (across {N_SEEDS} seeds)")

avg_within_subject_std = eegnet_subject_stds.mean()
print(f"\nAverage within-subject std across the {N_SEEDS} seeds: {avg_within_subject_std:.3f}  "
      f"-- this is a direct answer to 'how much does a single split's number move around'")

#Box plot: per-subject distribution of test accuracy across seeds
plt.figure(figsize=(10, 5))
plt.boxplot([within_subject_results[s]["eegnet_test_accs"] for s in subject_ids], labels=subject_ids)
plt.axhline(0.25, color="gray", linestyle=":", label="chance (0.25)")
plt.ylabel("Test balanced accuracy"); plt.xticks(rotation=45)
plt.title(f"Within-subject EEGNet test accuracy across {N_SEEDS} random 80/10/10 splits")
plt.legend(); plt.tight_layout(); plt.show()

#Topology maps: average highlight of learned filters from model 

median_acc = np.median(eegnet_subject_means)
representative_sub = subject_ids[int(np.argmin(np.abs(eegnet_subject_means - median_acc)))]
representative_model = within_subject_results[representative_sub]["model"]
print(f"\nUsing {representative_sub} (mean test acc {within_subject_results[representative_sub]['eegnet_test_accs'].mean():.3f} "
      f"across {N_SEEDS} seeds, closest to median {median_acc:.3f}) seed-0 model for interpretability plots.")

SAMPLE_INFO.set_montage("biosemi128", on_missing="warn")
eegnet_spatial_weights = representative_model.conv_spatial.weight.detach().cpu().numpy()[:, 0, :, 0]   # (16, 128)
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    plot_topomap(eegnet_spatial_weights[i], SAMPLE_INFO, axes=ax, show=False)
    ax.set_title(f"filter {i+1}", fontsize=9)
plt.suptitle(f"EEGNet ({representative_sub}, seed 0): all 16 learned spatial filters")
plt.tight_layout(); plt.show()
mean_importance = np.abs(eegnet_spatial_weights).mean(axis=0)
fig2, ax2 = plt.subplots(figsize=(5, 5))
im, _ = plot_topomap(mean_importance, SAMPLE_INFO, axes=ax2, show=False)
ax2.set_title(f"EEGNet ({representative_sub}, seed 0): mean |spatial weight| across all filters\n(overall electrode importance)")
plt.colorbar(im, ax=ax2); plt.show()

