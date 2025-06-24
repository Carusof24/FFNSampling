# data/load_dataset.py

import torch
from torch.utils.data import TensorDataset
import os

torch.set_default_dtype(torch.float64)

# Percorsi dei file nella cartella attuale
SAVE_DIR = os.path.expanduser("~/data")
OUT_DIM = 100
TRAIN_F = f"{SAVE_DIR}/fashion_train_{OUT_DIM}.pt"
TEST_F  = f"{SAVE_DIR}/fashion_test_{OUT_DIM}.pt"
PROJ_F  = f"{SAVE_DIR}/R_{OUT_DIM}.pt"

# ======== 1. Carica TRAIN e TEST proiettati =========
X_train, Y_train = torch.load(TRAIN_F, weights_only=True)
X_test,  Y_test  = torch.load(TEST_F,  weights_only=True)

# Converti in float64 per coerenza con la rete
X_train = X_train.double()
X_test  = X_test.double()

print("✔️ Dataset proiettati:")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ======== 2. Funzione per sottoinsieme del training set =========
def make_fashion_subset(P, seed=42):
    """Crea un sottoinsieme casuale del training set"""
    g   = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X_train), generator=g)[:P]
    return TensorDataset(X_train[idx], Y_train[idx])

