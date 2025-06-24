import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

from models.ffn import FeedforwardNet, NNModel
from utils.operations import Cost, Metric, init_torch_generator
from data.load_dataset import make_fashion_subset
from sampler.pl_tgpu import  PLSampler

# --------------------
# 1. Load model weights
# --------------------
device = 'cuda' 
weights_path = os.path.expanduser("~/weights/final_model_1.pt")

net = FeedforwardNet(input_dim=100, hidden_dim=100, output_dim=10).to(device)
model = NNModel(net, device=device)
model.load(weights_path)

# --------------------
# 3. Dataset with alpha scaling
# --------------------
alpha = 1.0
N = sum(p.numel() for p in model.NN.parameters() if p.requires_grad)
P = int(alpha * N)
dataset = make_fashion_subset(P, seed=42)
dataset.tensors = (dataset.tensors[0].to(device), dataset.tensors[1].to(device))
X_full, Y_full = dataset.tensors

Cost   = lambda logits, target: F.cross_entropy(logits, target)
Metric = lambda logits, target: (logits.argmax(dim=1) == target).float().mean()


# --------------------
# 5. Run Lang
# --------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = os.path.expanduser("~/results")

results_dir = os.path.join(base_dir, f"pl_runtg{timestamp}")
weights_dir = os.path.join(results_dir, "weights")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)




# Simulation blocks
stime_list    = [3.0e4, 2.0e4]        # simulated times (moves = stime/dt)
T_list        = [1.0e-6, 1.0e-6]    # target temperatures
Tratio_list   = [3.0e-1, 0.5e-1]    # ratio T_mb / T
dt_list       = [1,    1]       # integration time-step
m1_list       = [0.2,    0.7]       # mobility (c1 = sqrt(1 - m1^2))

# Static sampling parameters
pars = {
    'lamda':        1.0e-5,   # regularization coefficient
    'mbs':          64, # mini-batch size
    'max_extractions': 5000,  # max extractions for std estimate
    'extractions':     200,   # extractions per std cycle
    'threshold_est':  0.01,   # std convergence threshold
    'max_adj_step': 50000,    # max steps before re-adjourn
    'min_adj_step':  1000,    # base step for re-adjourn
    'opt_streak':      5,     # streak scale for adjustment
    'threshold_adj':  0.01,   # threshold for streak
    'seed':            0,     # RNG seed
    'lob':             False,
}

generator = init_torch_generator(seed=int(pars['seed']), device=device)
    # Define cost and metric

# Instantiate sampler
sampler = PLSampler(
    model=model,
    Cost=Cost,
    Metric=Metric,
    dataset=dataset,
    Y_full=Y_full,
    generator=generator,
    name=f'PL_sampler',
)


# Loop over each simulation block
for idx in range(len(stime_list)):
    # Dynamic block parameters
    stime  = stime_list[idx]
    T      = T_list[idx]
    Tratio = Tratio_list[idx]
    dt     = dt_list[idx]
    m1     = m1_list[idx]

    # Compute moves and temperatures
    moves       = int(stime / dt)
    pars['moves'] = moves
    pars['dt']    = dt
    pars['T']     = T
    pars['T_mb']  = Tratio * T
    pars['m1']    = m1


    # Define cost and metric


    # Determine save frequencies
    save_step  = 100
    check_step = 100
    wsave_step = max(1, moves // 100)
    print_step = 1000

    # Run sampling
    sampler.sample(
        pars=pars,
        results_dir=results_dir,
        weights_dir=weights_dir,
        prefixes={'w': 'w', 'm': 'm'},
        start=None,
        keep_going=False,
        save_step=save_step,
        check_step=check_step,
        wsave_step=wsave_step,
        print_step=print_step
    )

print("Sampling completed successfully.")

