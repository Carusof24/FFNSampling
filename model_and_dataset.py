# -*- coding: utf-8 -*-
"""Model and dataset.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o-HRbXHO0gsO7AaFNs67T43MrxBu1-em

# Feedfoward

VLa rete neurale feedforward definita è composta da un layer di input, due hidden layer e un layer di output. I due hidden layer, ciascuno con 100 neuroni, utilizzano la funzione di attivazione ReLU e Sigmoid nell’output con uscita di 10 classi. Il modello è completamente connesso - ogni neurone di un layer sia collegato a tutti i neuroni del layer successivo
"""

import torch
import torch.nn as nn

# Default precision
torch.set_default_dtype(torch.float64)

# -------------------- #
# Feedforward NN Model #
# -------------------- #
class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100, output_dim=10):
        super(FeedforwardNet, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Secondo hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Layer di output
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        #these are fully connected layers
        # Activaction function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)   # Attivazione ReLU sul primo hidden layer
        x = self.fc2(x)
        x = self.relu(x)   # Attivazione ReLU
        x = self.fc3(x)
        x = self.sigmoid(x)  # Sigmoid per l'output
        return x

#wrapper is a class that encapsulates another class
# NNModel, this class provides a wrapper around the FFN class to manage its weights
class NNModel():
    def __init__(self, NN, device='cpu', f=None):
        self.NN = NN
        self.device = device if ('cuda' in device) and torch.cuda.is_available() else 'cpu'
        if f:
            self.load(f)
        else:
            self._to_device()
            self._init_weights()

    def _init_weights(self):
        self.weights = {name: param for name, param in self.NN.named_parameters() if param.requires_grad}

    def copy(self, grad=False):
        if not grad:
            wcopy = {name: self.weights[name].detach().clone() for name in self.weights}
        else:
            wcopy = {name: self.weights[name].grad.detach().clone() for name in self.weights}
        return wcopy

    def set_weights(self, wnew):
        assert all(name in self.weights for name in wnew), f"NNModel.set_weights(): invalid layer found in wnew. Allowed values: {list(self.weights.keys())}"
        for name, new_param in wnew.items():
            for pname, param in self.NN.named_parameters():
                if pname == name:
                    param.data = new_param.detach().clone()
        self._init_weights()

    def load(self, f):
        with open(f, 'rb') as ptf:
            self.NN.load_state_dict(torch.load(ptf, map_location=torch.device(self.device)))
        self._to_device()
        self._init_weights()

    def save(self, f):
        with open(f, 'wb') as ptf:
            torch.save(self.NN.state_dict(), ptf)

    def _to_device(self):
        if 'cuda' in self.device:
            self.NN.to(self.device)

import torch
import torch.nn as nn

# Default precision
torch.set_default_dtype(torch.float64)

# -------------------- #
#     Feedforward.     #
# -------------------- #
class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100, output_dim=10):
        super(FeedforwardNet, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Output layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # Sigmoid per l'output
        return x


# ---------------------- #
       # NNModel #
# ---------------------- #
class NNModel():
    def __init__(self, NN, device='cpu', f=None):
        self.NN = NN
        self.device = device if ('cuda' in device) and torch.cuda.is_available() else 'cpu'
        if f:
            self.load(f)
        else:
            self._to_device()
            self._init_weights()

    def _init_weights(self):
        self.weights = {name: param for name, param in self.NN.named_parameters() if param.requires_grad}

    def copy(self, grad=False):
        if not grad:
            wcopy = {name: self.weights[name].detach().clone() for name in self.weights}
        else:
            wcopy = {name: self.weights[name].grad.detach().clone() for name in self.weights}
        return wcopy

    def set_weights(self, wnew):
        assert all(name in self.weights for name in wnew), f"NNModel.set_weights(): invalid layer found in wnew. Allowed values: {list(self.weights.keys())}"
        for name, new_param in wnew.items():
            module_name, param_name = name.rsplit('.', 1)  # Divide il nome del modulo e del parametro
            module = self.NN.get_submodule(module_name)  # Ottiene il submodulo
            setattr(module, param_name, nn.Parameter(new_param.clone().detach()))  # Imposta i nuovi pesi
        self._init_weights()

    def load(self, f):
        with open(f, 'rb') as ptf:
            self.NN.load_state_dict(torch.load(ptf, map_location=torch.device(self.device)))
        self._to_device()
        self._init_weights()

    def save(self, f):
        with open(f, 'wb') as ptf:
            torch.save(self.NN.state_dict(), ptf)

    def _to_device(self):
        if 'cuda' in self.device:
            self.NN.to(self.device)

import torch

# Inizializza la rete e il wrapper
net = FeedforwardNet()
model = NNModel(net)

# Salvataggio dei pesi iniziali
w_initial = model.copy()

# Stampa dei pesi iniziali del layer fc1
print("Pesi iniziali di fc1.weight:")
print(model.weights['fc1.weight'])

# Simulazione di una mossa, modifica del peso fc1.weight
wnew = model.copy()
wnew['fc1.weight'] = wnew['fc1.weight'] + 1.0  # Aggiunge 1 a ogni peso

# Applica la mossa proposta
model.set_weights(wnew)
print("\nPesi dopo la mossa proposta di fc1.weight:")
print(model.weights['fc1.weight'])

# Supponiamo di voler rifiutare la mossa: ripristino della configurazione precedente
model.set_weights(w_initial)
print("\nPesi dopo il ripristino (rifiuto della mossa) di fc1.weight:")
print(model.weights['fc1.weight'])

#Manage e manipulate the weights of neural network using NNModel.
#Make changes to the weights (simulating a "move" in a search or optimization process).
#Revert changes to the weights if needed (like rejecting a move).
# Inizializza la rete e il wrapper
net = FeedforwardNet()
model = NNModel(net)

# Salvataggio dei pesi iniziali
w_initial = model.copy()

# Stampa dei pesi iniziali del layer fc1
print("Pesi iniziali di fc1.weight:")
print(model.weights['fc1.weight'])

# Simulazione di una mossa, modifica del peso fc1.weight
wnew = model.copy()
wnew['fc1.weight'] = wnew['fc1.weight'] + 1.0

# Applica la mossa proposta
model.set_weights(wnew)
print("\nPesi dopo la mossa proposta di fc1.weight:")
print(model.weights['fc1.weight'])

# Supponiamo di voler rifiutare la mossa: ripristino della configurazione precedente
model.set_weights(w_initial)
print("\nPesi dopo il ripristino (rifiuto della mossa) di fc1.weight:")
print(model.weights['fc1.weight'])

##Creiamo un dizionario dei pesi della rete, e specificamente i pesi del primo layer (fc1.weight) vengono sostituiti con valori casuali generati da una distribuzione normale.
#I pesi modificati vengono quindi reinseriti nel modello tramite il metodo set_weights.
#Il codice assicura che i pesi aggiornati nel modello corrispondano a quelli salvati nel wrapper NNModel.
#si verifica che i pesi attuali della rete coincidano con quelli memorizzati nel dizionario dei pesi.
#La verifica è effettuata confrontando i pesi presenti nel modello e quelli nel dizionario dopo il reset.
#il codice dimostra come gestire la modifica, il salvataggio e il ripristino dei pesi in una rete neurale implementata
# # Feedfoward

# Default precision
torch.set_default_dtype(torch.float64)

# parameters:
input_dim = 100
hidden_dim = 100
output_dim = 10

# Initialize the neural network
net = FeedforwardNet(input_dim, hidden_dim, output_dim)
model = NNModel(net)


# Get the initial weights
initial_weights = model.copy()


# Modify the weights
modified_weights = model.copy()

# change the weight of the first layer
for name, param in modified_weights.items():
    if name == 'fc1.weight':
        param.data = torch.randn_like(param.data)
        break


# Set the modified weights back into the model
model.set_weights(modified_weights)


# Verify that the weights in the NN and the weights stored in model.weights are the same
for name, param in model.NN.named_parameters():
    if name in modified_weights:
      print(f"Layer: {name}")
      print("Difference between model.weights and NN parameters:", torch.equal(model.weights[name], param.data))

# Reset weights to initial values (example)
model.set_weights(initial_weights)

# Verify that the weights have been reset correctly
for name, param in model.NN.named_parameters():
    if name in initial_weights:
      print(f"Layer: {name}")
      print("Difference between model.weights and NN parameters after reset:", torch.equal(model.weights[name], param.data))

"""##Dataset

"""

import numpy as np
import torch
from torch.utils.data import Dataset

def flip_spin(vector, p=0.1):
    """
    Data una sequenza di spin (±1), inverte ciascun elemento con probabilità p.
    """
    flip_mask = np.random.rand(vector.shape[0]) < p  # Indica quali spin invertire
    vector_flipped = vector.copy()
    vector_flipped[flip_mask] *= -1  # Applica il flip agli spin selezionati
    return vector_flipped

def generate_dataset(P, K=10, vector_dim=100, p_flip=0.1, seed=0):
    """
    Genera un dataset di P esempi distribuiti su K classi.

    1. Crea K vettori base indip, uno per classe.
    2. Estrai P esempi campionando una classe k e applicando il flip allo spin.
    """
    np.random.seed(seed)  #

    # Step 1: Generiamo i K vettori base, uno per ogni classe
    ref_vectors = [np.random.choice([-1, 1], size=vector_dim) for _ in range(K)]

    # Step 2: Campioniamo P esempi assegnando loro una classe casuale e applicando il flip
    labels, vectors = [], []
    for _ in range(P):
        k = np.random.randint(0, K)  # Scegli una classe casuale
        flipped_vector = flip_spin(ref_vectors[k], p_flip)  # Applica il flip al vettore della classe k
        vectors.append(flipped_vector)

        label = np.zeros(K)
        label[k] = 1  # One-hot encoding della classe
        labels.append(label)

    return np.array(vectors), np.array(labels)

class SpinDataset(Dataset):
    """
    Dataset compatibile con PyTorch, che restituisce coppie (input, target).
    """
    def __init__(self, P, vector_dim=100, K=10, p_flip=0.1, seed=0):
        X, y = generate_dataset(P, K, vector_dim, p_flip, seed)
        self.X = torch.tensor(X, dtype=torch.float32)  # Input
        self.y = torch.tensor(y, dtype=torch.float32)  # One-hot label
        self.P = P

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# AVRo SBAGliato anche stavolta
if __name__ == "__main__":
    P = 100  # Datasize del dataset
    K = 10
    dataset = SpinDataset(P=P, vector_dim=100, K=K, p_flip=0.1, seed=0)

    print("Numero totale di esempi nel dataset:", len(dataset))

    # Stampa un esempio per classe
    for cls in range(K):
        examples = [ (x,y) for x,y in dataset if torch.argmax(y).item() == cls ]
        print(f"Classe {cls}: {len(examples)} esempi")

import numpy as np

# Parameters
P, K, vector_dim, p_flip, seed = 100, 10, 100, 0.1, 0

# Genera il dataset )
vectors, labels = generate_dataset(P, K, vector_dim, p_flip, seed)

# Ricostruisci i vettori di riferimento (stesso seed usato in generate_dataset)
np.random.seed(seed)
ref_vectors = [np.random.choice([-1, 1], size=vector_dim) for _ in range(K)]

# Verifica per una classe specifica, ad esempio classe 0
class_to_check = 0
ref = ref_vectors[class_to_check]

# Calcola la percentuale di flip per ogni esempio della classe scelta
flip_percents = [
    np.sum(vec != ref) / vector_dim
    for vec, lab in zip(vectors, labels)
    if np.argmax(lab) == class_to_check
]

print("Media flip % per classe {}: {:.2f}%".format(class_to_check, 100 * np.mean(flip_percents)))
print("Massimo flip % per classe {}: {:.2f}%".format(class_to_check, 100 * np.max(flip_percents)))

#HO UN PROBLEMA PERCHE LA CLASSE 5 HA UN MASSIMO DI FLIP AL 23% è l'unicaaaaaaaa



"""# training with ADAM

# training with SGD
"""

