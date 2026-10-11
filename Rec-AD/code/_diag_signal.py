"""Diagnostic: is the sequence/graph signal absent from the data, or lost in
the model readout?

Run:  python _diag_signal.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from data.contract import load_contract
from data.dataset import build_loaders, build_spec, loader_arrays
from evaluate import compute_metrics

data = load_contract("data_cache/_selftest.npz")
spec = build_spec(data)
print("spec:", spec)
print()

loaders = build_loaders(data, batch_size=512, num_workers=0)

# --- 1. raw signal in node_feat, straight from numpy (no model involved) ---
_, _, y, at = loader_arrays(loaders["test"])
nf = data["node_feat"]
idx = loaders["test"].dataset.index
nf_test = nf[idx]

mean_pool = nf_test.mean(axis=1)          # (N, F)
max_pool = np.abs(nf_test).max(axis=1)    # (N, F)
std_pool = nf_test.std(axis=1)            # (N, F)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def probe(name, X):
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs, y)
    s = clf.predict_proba(Xs)[:, 1]
    print(f"  {name:<34} AUC={compute_metrics(y, s)['auc']:.4f}")

print("Numpy probes on node_feat (memorisation upper bound):")
probe("mean pool only", mean_pool)
probe("max|.| pool only", max_pool)
probe("std pool only", std_pool)
probe("flattened (all 14 buses)", nf_test.reshape(len(nf_test), -1))
print()

# --- 2. what the pooled readout can see ---
print("Pooled-feature separability (this is what the models' heads receive):")
for label, arr in (("mean", mean_pool), ("max", max_pool)):
    for cls, name in ((0, "normal"), (1, "attacked")):
        m = arr[y == cls].mean(axis=0)
        print(f"  {label:<5} {name:<9} feature means: {np.round(m, 3)}")
print()

# --- 3. does the LSTM learn at all? ---
from baselines.lstm import LSTMDetector
from train import run_training_loop

cfg = {"train": {"epochs": 25, "lr": 1e-3, "weight_decay": 1e-4,
                 "early_stop_patience": 50, "grad_clip": 5.0},
       "baselines": {"lstm": {"hidden": 64, "layers": 2}}}

print("Training LSTM for 25 epochs with a mean+max readout probe...")
model = LSTMDetector(spec, cfg, name="LSTM")
hist = run_training_loop(model, loaders["train"], loaders["val"],
                         device=torch.device("cpu"))
scores = model.predict_proba(loaders["test"])
print(f"  LSTM  test AUC = {compute_metrics(y, scores)['auc']:.4f}")
print(f"  train_loss first={hist['train_loss'][0]:.4f} "
      f"last={hist['train_loss'][-1]:.4f}")
print(f"  score range: min={scores.min():.4f} max={scores.max():.4f} "
      f"std={scores.std():.4f}")
