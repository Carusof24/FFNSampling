import torch

# Default precision
torch.set_default_dtype(torch.float64)


# -------------------- #
# Neural Network Model #
# -------------------- #
class NNModel():

    # Initialization neural network model
    def __init__(
            self,
            NN,
            device='cpu',
            f=None,
    ):
        self.NN = NN
        self.device = device if ('cuda' in device) and torch.cuda.is_available() else 'cpu'
        if f:
            self.load(f)
        else:
            self._to_device()
            self._init_weights()

    # Initialization of the NN weights dictionary
    def _init_weights(self):
        self.weights = {
                name: param for name, param in self.NN.named_parameters() if param.requires_grad
        }

    # Returns a copy of the NN weights (or gradient)
    def copy(self, grad=False):
        if not grad:
            wcopy = {name: self.weights[name].detach().clone() for name in self.weights}
        else:
            wcopy = {name: self.weights[name].grad.detach().clone() for name in self.weights}
        return wcopy

    # Set NN weights
    def set_weights(self, wnew):
        assert all([name in self.weights.keys() for name in wnew]), f"NNModel.set_weights(): invalid layer found in wnew. Allowed values: {self.weights.keys()}"
        for wname in wnew:
            for pname, param in self.NN.named_parameters():
                if wname != pname: continue
                param.data = wnew[wname].detach().clone().requires_grad_(True)
        self._init_weights()

    # Load NN weights from file
    def load(self, f):
        with open(f, 'rb') as ptf:
            self.NN.load_state_dict(torch.load(ptf, map_location=torch.device(self.device)))
        self._to_device()
        self._init_weights()

    # Save NN weights to file
    def save(self, f):
        with open(f, 'wb') as ptf:
            torch.save(self.NN.state_dict(), ptf)

    # Transfer NN weights to device
    def _to_device(self):
        if 'cuda' in self.device:
            self.NN.to(self.device)




from ffn import FeedforwardNet, NNModel
import torch

torch.set_default_dtype(torch.float64)
torch.manual_seed(42)

net = FeedforwardNet(input_dim=100, hidden_dim=100, output_dim=10)
model = NNModel(net, device='cuda')

N = sum(p.numel() for p in model.NN.parameters() if p.requires_grad)
print(f"Calcolato N: {N}")

