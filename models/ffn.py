# models/feedforward_net.py

import torch
import torch.nn as nn

# Feedforward Neural Network
class FeedforwardNet(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=100, output_dim=10):
        super(FeedforwardNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Wrapper per gestire pesi, salvataggi, copia ecc.
class NNModel:
    def __init__(self, NN, device='cpu', f=None):
        self.NN = NN
        self.device = device if ('cuda' in device) and torch.cuda.is_available() else 'cpu'
        if f:
            self.load(f)
        else:
            self._to_device()
            self._init_weights()

    def __call__(self, x):
        return self.NN(x)

    def zero_grad(self):
        for p in self.NN.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def _init_weights(self):
        self.weights = {name: param for name, param in self.NN.named_parameters() if param.requires_grad}

    def copy(self, grad=False):
        if not grad:
            wcopy = {name: self.weights[name].detach().clone() for name in self.weights}
        else:
            wcopy = {name: self.weights[name].grad.detach().clone() for name in self.weights}
        return wcopy

    def set_weights(self, wnew):
        for name, new_param in wnew.items():
            if name in self.weights:
                for pname, param in self.NN.named_parameters():
                    if pname == name:
                        param.data = new_param.detach().clone()
        self._init_weights()

    def load(self, f):
        with open(f, 'rb') as ptf:
            self.NN.load_state_dict(torch.load(ptf, map_location=torch.device(self.device), weights_only=True))
        self._to_device()
        self._init_weights()

    def save(self, f):
        with open(f, 'wb') as ptf:
            torch.save(self.NN.state_dict(), ptf)

    def _to_device(self):
        self.NN.to(self.device)

