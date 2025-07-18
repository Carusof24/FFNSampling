{\rtf1\ansi\ansicpg1252\cocoartf2512
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import torch\
from torch.utils.data import Dataset\
\
def flip_spin(vector, p=0.1, seed=4):\
    """\
    Data una sequenza di spin (\'b11), inverte ciascun elemento con probabilit\'e0 p.\
    """\
    flip_mask = np.random.rand(vector.shape[0]) < p  # Indica quali spin invertire\
    vector_flipped = vector.copy()\
    vector_flipped[flip_mask] *= -1  # Applica il flip agli spin selezionati\
    return vector_flipped\
\
def generate_dataset(P, K=10, vector_dim=100, p_flip=0.1, seed=0):\
    """\
    Genera un dataset di P esempi distribuiti su K classi.\
\
    1. Crea K vettori base indip, uno per classe.\
    2. Estrai P esempi campionando una classe k e applicando il flip allo spin.\
    """\
    np.random.seed(seed)\
\
    # Step 1: Generiamo i K vettori base, uno per ogni classe\
    ref_vectors = [np.random.choice([-1, 1], size=vector_dim) for _ in range(K)]\
\
    # Step 2: Campioniamo P esempi assegnando loro una classe casuale e applicando il flip\
    labels, vectors = [], []\
    for _ in range(P):\
        k = np.random.randint(0, K)  # Scegli una classe casuale\
        flipped_vector = flip_spin(ref_vectors[k], p_flip)  # Applica il flip al vettore della classe k\
        vectors.append(flipped_vector)\
\
        label = np.zeros(K)\
        label[k] = 1  # One-hot encoding della classe\
        labels.append(label)\
\
    return np.array(vectors), np.array(labels)\
\
class SpinDataset(Dataset):\
    """\
    Dataset compatibile con PyTorch, che restituisce coppie (input, target).\
    """\
    def __init__(self, P, vector_dim=100, K=10, p_flip=0.1, seed=4):\
        X, y = generate_dataset(P, K, vector_dim, p_flip, seed)\
        self.X = torch.tensor(X, dtype=torch.float64)  # Input\
        self.y = torch.tensor(y, dtype=torch.float64)  # One-hot label\
        self.P = P\
\
    def __len__(self):\
        return len(self.X)\
\
    def __getitem__(self, idx):\
        return self.X[idx], self.y[idx]\
\
def verify_class_distribution(dataset, K=10):\
    """\
    Verifica la distribuzione degli esempi per classe nel dataset.\
    Restituisce un array con il numero di esempi per ciascuna classe.\
    """\
    labels = dataset.y.numpy()\
    class_indices = np.argmax(labels, axis=1)\
    counts = np.zeros(K, dtype=int)\
    for cls in class_indices:\
        counts[cls] += 1\
    return counts\
\
# Test del dataset\
if __name__ == "__main__":\
    P = 1000  # Dimensione totale del dataset\
    K = 10\
    dataset = SpinDataset(P=P, vector_dim=100, K=K, p_flip=0.1, seed=4)\
\
    print("Numero totale di esempi nel dataset:", len(dataset))\
    counts = verify_class_distribution(dataset, K)\
    print("Distribuzione per classe (esempi per classe):", counts)\
\
    # Stampa un esempio per classe per verificare\
    for cls in range(K):\
        examples = [ (x,y) for x,y in dataset if torch.argmax(y).item() == cls ]\
        print(f"Classe \{cls\}: \{len(examples)\} esempi")}