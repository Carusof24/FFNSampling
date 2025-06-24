
# *** SAMPLER QUA SOTTO é stimo STD 1layer ****# %%
import os, shutil
import torch
import numpy as np
from time import perf_counter as ptime
import torch.nn.functional as F


from utils.operations import compute_d, compute_d2,  compute_mod2, wcopy, wsum, kpow, load_torch_state, save_torch_state, init_torch_generator, compute_mod, compute_q
# Default precision
torch.set_default_dtype(torch.float64)






# ------------------------------------ #
# Double-Noise Pseudo-Langevin Sampler #
# ------------------------------------ #
class PLSampler():

    ## Initialize the sampler
    #
    def __init__(self, model, Cost, Metric, dataset, Y_full, generator, name:str='PLSampler'):
        self.model = model
        self.Cost = Cost
        self.Metric = Metric
        self.dataset = dataset
        self.Y_full = Y_full # s
        self.generator = generator
        self.name = name
        # Initialize w0 here since it's used in _check_detailed_balance
        self.w0 = self.model.copy(grad=False)



    ## Initialize check detailed balance dictionary
    def _init_check(self, move):
        self.check = {
                'move': move,
                'logM': 0.,
                'dU': 0.,
                'dH': 0.,
                'wratio': -1.,
                'K': 0.,
                'd2': 0.,
                'dK': 0.,
                'btb_d': 0.,
                'dt_move': 0.,
                'eff_v': 0.,
                # Initialize eff_D and similarity here
                'eff_D': 0.,
                'similarity': 1.,
        }


    ## Perform double-noise pseudo-Langevin sampling
    def sample(
            self,
            pars, results_dir, weights_dir, prefixes,
            start=None, keep_going=False,
            save_step=10, check_step=10, wsave_step=1000, print_step=1000,
    ):
        self.pars = pars.copy()
        self.results_dir = results_dir
        self.weights_dir = weights_dir
        self.prefixes = prefixes
        if self.pars['lob']:
            self.pars['lamda'] *= self.pars['T']

        # Assuming diff_layers are the layers with learnable parameters

        if not hasattr(self.model, 'diff_layers'):
             self.model.diff_layers = list(self.model.weights.keys())
        if not hasattr(self, 'shapes'):
             self.shapes = {layer: self.model.weights[layer].shape for layer in self.model.diff_layers}

        if torch.cuda.is_available():
            #  sposta modello e tensori interni sulla prima GPU
            self.model.NN.to('cuda')
            self.model.device = torch.device('cuda')
            if self.generator.device.type != self.model.device.type:
                 self.generator = init_torch_generator(seed=int(self.pars['seed']), device=self.model.device) # Re-initialize generator on correct device
        elif 'cuda' in self.model.device.type:
             # Raise error only if sampler was configured for cuda but not available
             raise RuntimeError("CUDA non disponibile ma il sampler è configurato per usare la GPU.")

        if start is None:
            if not keep_going:
                self.t0 = ptime()
                self._adjourn_data(0)
                self._init_check(0)
            self._adjourn_pars(self.data['move'])
            self.momenta = {
                    layer: torch.randn(self.shapes[layer], device=self.model.device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m'])
                    for layer in self.model.diff_layers
            }
        else:
            self.t0 = ptime()-start['data']['time']
            self.data = start['data'].copy()
            for key in start['vpars']:
                self.pars[key] = start['vpars'][key]
            self.vpars = start['vpars'].copy()
            self.check = start['check'].copy()
            self.model.load(start['files']['model'])

            


            # Check if momenta file exists before loading
            if os.path.exists(start['files']['momenta']):
                self.momenta = torch.load(start['files']['momenta'], map_location=torch.device(self.model.device), weights_only=True)
            else:
                 # Initialize momenta if file not found (e.g., first run)
                 self.momenta = {
                     layer: torch.randn(self.shapes[layer], device=self.model.device, generator=self.generator) * np.sqrt(self.pars['T']*self.pars['m'])
                     for layer in self.model.diff_layers
                 }
                 print(f"Warning: Momenta file not found at {start['files']['momenta']}. Initializing momenta randomly.")

            # Check if generator state file exists before loading
            if os.path.exists(start['files']['generator']):
                 self.generator = load_torch_state(start['files']['generator'], self.generator)
            else:
                 # Re-initialize generator if file not found
                 self.generator = init_torch_generator(seed=int(self.pars['seed']), device=self.model.device)
                 print(f"Warning: Generator state file not found at {start['files']['generator']}. Re-initializing generator.")

        self._to_types()
        
        



        if self.data['move'] == 0:
            if save_step > 0:
                self._save(self.data, f'{self.results_dir}/data.dat', header=True)
                self._save(self.vpars, f'{self.results_dir}/vpars.dat', header=True)
            if check_step > 0: self._save(self.check, f'{self.results_dir}/check.dat', header=True)
            if wsave_step > 0: self._wsave(0)
        if print_step > 0:
            self._prepare_printing()
            self._status()
        

        assert torch.cuda.is_available(), "Questo sampler richiede una GPU CUDA"

        # ------------------------------------------------------------------
        # pre-loop: stream ed eventi riutilizzabili ogni mossa
        # ------------------------------------------------------------------
        if 'cuda' in self.model.device.type:
            stream = torch.cuda.Stream(device=self.model.device)

            e_move_s = torch.cuda.Event(enable_timing=True)
            e_move_e = torch.cuda.Event(enable_timing=True)
            e_pl_s   = torch.cuda.Event(enable_timing=True)
            e_pl_e   = torch.cuda.Event(enable_timing=True)

        for move in range(self.data['move'], self.data['move'] + self.pars['moves']):

            # --------------------------------------------------------------
            # inizio mossa – cronometro totale GPU
            # --------------------------------------------------------------
            with torch.cuda.stream(stream):
                e_move_s.record(stream)

            # ---------- inizializza contatori locali ----------
            dt_pl = dt_adj = dt_bal = 0.0

            # =============================================================
            #                       A D J O U R N
            # =============================================================
            if (move + 1) % self.pars['adj_step'] == 0:
                t0 = ptime()
                self._adjourn_pars(move + 1)
                self._save(self.vpars, f'{self.results_dir}/vpars.dat')
                dt_adj += ptime() - t0          # CPU / I-O → wall-time

            # =============================================================
            #                       C H E C K
            # =============================================================
            if (move + 1) % check_step == 0:

                # -------- integrazione (GPU) --------
                with torch.cuda.stream(stream):
                    e_pl_s.record(stream)
                    old_weights, old_K = self.model.copy(grad=False), self._compute_K()
                    self._step()                 
                    e_pl_e.record(stream)

                # -------- salvataggio dati ---------
                if (move + 1) % save_step == 0:
                    t0 = ptime()
                    self._adjourn_data(move + 1)
                    self._save(self.data, f'{self.results_dir}/data.dat')
                    dt_adj += ptime() - t0
                 

                # -------- detailed balance ---------
                t0 = ptime()                       # CPU-side check
                dt_pl_gpu_ms = e_pl_s.elapsed_time(e_pl_e)  # 
                dt_tot = (dt_pl_gpu_ms / 1000.0) + dt_adj   # GPU + I/O
                self._check_detailed_balance(move + 1, old_weights, old_K, dt_tot)
                self._save(self.check_bal, f'{self.results_dir}/check_DB.dat')
                dt_bal = ptime() - t0

                del old_weights

            # =============================================================
            #                       S A V E   
            # =============================================================
            elif (move + 1) % save_step == 0:

                # -------- integrazione (GPU) --------
                with torch.cuda.stream(stream):
                    e_pl_s.record(stream)
                    self._step()
                    e_pl_e.record(stream)

                # -------- salvataggio dati ---------
                t0 = ptime()
                self._adjourn_data(move + 1)
                self._save(self.data, f'{self.results_dir}/data.dat')
                dt_adj += ptime() - t0

            # =============================================================
            #                       M O V E  
            # =============================================================
            else:
                with torch.cuda.stream(stream):
                    e_pl_s.record(stream)
                    self._step()
                    e_pl_e.record(stream)

            # --------------------------------------------------------------
            # fine mossa – chiudi cronometro GPU
            # --------------------------------------------------------------
            with torch.cuda.stream(stream):
                e_move_e.record(stream)

            stream.synchronize()

            # ms → s
            dt_pl  = e_pl_s.elapsed_time(e_pl_e) / 1000.0    # integrazione pura
            dt_move = e_move_s.elapsed_time(e_move_e) / 1000.0  # mossa intera

            self._check_data(self, move+1, dt_move)
            self._save(self.check_data, f'{self.results_dir}/check_runtime.dat')

            # ============== aggiorna statistiche ===============
            self.check['dt_pl']  = dt_pl
            self.check['dt_adj'] = dt_adj
            self.check['dt_bal'] = dt_bal
            self.check['dt_move'] = dt_move

            #------------------
            self.check['eff_D'] = self.check['d2'] / (dt_move * (move + 1))
            self.check['eff_v'] = self.check['btb_d'] / dt_move

            # -------  -------
            # 
            if wsave_step > 0 and (move+1) % wsave_step == 0:
                self._wsave(move+1)
            if print_step > 0 and (move+1) % print_step == 0:
                self._status()



    ## Integration step (velocity Verlet algorithm applied to Langevin equations)
    def _step(self):
        old_grad = self._compute_grad()
        old_noise = self._generate_noise()

        with torch.no_grad():
            for layer in self.model.diff_layers:
                self.model.weights[layer] += self.momenta[layer]*self.pars['c1']*self.pars['dt']/self.pars['m'] - old_grad[layer]*self.pars['dt']**2./(2.*self.pars['m']) + old_noise[layer]*self.pars['k_wn']*self.pars['dt']/self.pars['m']

        new_grad = self._compute_grad()
        new_noise = self._generate_noise()

        for layer in self.model.weights:
            self.momenta[layer] = self.momenta[layer]*self.pars['c1']**2. - (new_grad[layer]+old_grad[layer])*self.pars['c1']*self.pars['dt']/2. + old_noise[layer]*self.pars['c1']*self.pars['k_wn'] + new_noise[layer]*np.sqrt((1.-self.pars['c1']**2.)*self.pars['k_mb']**2. + self.pars['k_wn']**2.)


   ## Compute an estimate of the current standard deviation std deriving from the mini-batch extraction
    def _estimate_std(self):
        sum_grad, sum2_grad = {}, {}
        for iext in range(self.pars['extractions']):
            grad = self._compute_grad()
            if iext == 0:
                for layer in grad:
                    sum_grad[layer] = grad[layer].detach().clone()
                    sum2_grad[layer] = (grad[layer]**2.).detach().clone()
            else:
                for layer in grad:
                    sum_grad[layer] += grad[layer].detach().clone()
                    sum2_grad[layer] += (grad[layer]**2.).detach().clone()
        tot_extractions = self.pars['extractions']

        # Assuming the first layer is representative for std estimation
        #
        first_layer_name = list(sum_grad.keys())[0]
        prev_std = torch.sqrt(( sum2_grad[first_layer_name]/(tot_extractions) - (sum_grad[first_layer_name]/(tot_extractions))**2. ).mean()).item()
        """

        """

        while tot_extractions < self.pars['max_extractions']:
            for _ in range(self.pars['extractions']):
                grad = self._compute_grad()
                for name in grad:
                    sum_grad[name] += grad[name].detach().clone()
                    sum2_grad[name] += (grad[name]**2.).detach().clone()
            tot_extractions += self.pars['extractions']
            # Recalculate std based on the same representative layer
            curr_std = torch.sqrt((sum2_grad[first_layer_name]/(tot_extractions) - (sum_grad[first_layer_name]/(tot_extractions))**2.).mean()).item()
            convergence = abs(curr_std-prev_std)/prev_std <= self.pars['threshold_est']
            prev_std = curr_std
            if convergence:
                break

        return prev_std


    ## Adjourn the current value of the mini-batch noise and check whether to increase the step
    def _adjourn_pars(self, move):
        curr_std = self._estimate_std()
        if 'std' not in self.pars.keys():
            self.pars['streak'] = 1
            self.pars['c1'] = np.sqrt(1.-self.pars['m1']**2.)
            # Ensure self.pars['m'] is not zero before division
            if self.pars['m1'] == 0:
                # Handle this case appropriately, perhaps set m to a large value or raise an error
                print("Warning: m1 is 0, which would lead to division by zero in calculating 'm'. Adjusting m.")
                self.pars['m'] = 1e6 # Example: set to a large value
            else:
                 self.pars['m'] = (self.pars['dt']*curr_std)**2./(4.*self.pars['T_mb']*self.pars['m1']**2.)
        else:
            keep_streak = abs(curr_std-self.pars['std'])/self.pars['std'] <= self.pars['threshold_adj']
            self.pars['streak'] = self.pars['streak']+1 if keep_streak else 1
        self.pars['adj_step'] = round( np.tanh(self.pars['streak']/self.pars['opt_streak']) * self.pars['max_adj_step']/self.pars['min_adj_step'] ) * self.pars['min_adj_step']
        self.pars['std'] = curr_std

        # Ensure self.pars['m'] is not zero before division
        if self.pars['m'] == 0 or self.pars['m1'] == 0:
             print("Warning: m or m1 is 0. Skipping update for T_mb, k_mb, k_wn.")
             # You might want to handle this case differently depending on the desired behavior
        else:
            self.pars['T_mb'] = (self.pars['dt']*self.pars['std'])**2./(4.*self.pars['m']*self.pars['m1']**2.)
            self.pars['k_mb'] = self.pars['dt']*self.pars['std']/2.
            # Ensure self.pars['T_mb'] is not zero before division and T >= T_mb
            if self.pars['T_mb'] == 0 or self.pars['T'] < self.pars['T_mb']:
                 print("Warning: T_mb is 0 or T < T_mb. Skipping update for k_wn.")
                 # Handle this case
            else:
                 self.pars['k_wn'] = self.pars['k_mb'] * np.sqrt((self.pars['T']-self.pars['T_mb'])/self.pars['T_mb'])

        self.vpars = {'move': move}
        for key in ('std', 'T_mb', 'k_mb', 'k_wn', 'streak', 'adj_step'):
            self.vpars[key] = self.pars.get(key, None) # Use .get() to avoid KeyError if key wasn't set


    ## Extract random mini-batch and return gradient computed on it
    def _compute_grad(self):
        mb_idxs = torch.zeros((len(self.dataset),), dtype=torch.bool)
        while mb_idxs.sum() < self.pars['mbs']:
            idx = torch.randint(low=0, high=len(self.dataset), size=(1,), device=self.generator.device, generator=self.generator)
            mb_idxs[idx] = True
        # Access mini-batch data and labels using TensorDataset indexing
        x, y = self.dataset[mb_idxs]
        x = x.to(self.model.device)
        y = y.to(self.model.device)
        _ = self._compute_observables(x, y, backward=True, extra=False)
        grad = self.model.copy(grad=True)
        self.model.zero_grad()
        return grad


    ## Calculate observables (
    #
    def _compute_observables(self, x, y, backward=True, extra=False):
        fx = self.model(x)
        cost = self.Cost(fx, y)
        mod2 = compute_mod2(self.model.weights)
        loss = cost + (self.pars['lamda']/2.)*mod2

        if backward:
            loss.backward(retain_graph=True)

        if extra: ### a
            #
            full_x, full_y = self.dataset[:]
            full_x = full_x.to(self.model.device)
            full_y = full_y.to(self.model.device) #
            full_fx = self.model(full_x)
            metric = self.Metric(full_fx, full_y)
            return {'loss':loss, 'cost':cost, 'mod2':mod2, 'metric':metric}
        else:
            return {'loss':loss, 'cost':cost, 'mod2':mod2}


    ## Generate true white noise
    def _generate_noise(self):
        noise = {
                layer: torch.randn(self.shapes[layer], device=self.model.device, generator=self.generator)
                for layer in self.model.diff_layers
        }
        return noise


    ## Adjourn data (computing observables et)
    # s
    def _adjourn_data(self, move):
        self.data = {'move':move}
        full_x, full_y = self.dataset[:] # Access all data from the TensorDataset
        observables = self._compute_observables(full_x, full_y, backward=False, extra=True)
        for key in observables:
            self.data[key] = observables[key].item()
        self.data['time'] = ptime()-self.t0





    ## Compute the value of the kinetic energy
    def _compute_K(self):
        return 1./(2.*self.pars['m']) * compute_mod2(self.momenta).item()


 ## Check the validity of the detailed balance for gaussian-distributed noise
    def _check_detailed_balance(self, move, old_weights, old_K, dt_move):




        full_x, full_y = self.dataset[:]
        new_potential = self._compute_observables(full_x, full_y, backward=True, extra=False)['loss']
        new_grad      = self.model.copy(grad=True)

        # Ripristino pesi vecchi per calcolare old_potential
        new_weights = self.model.copy(grad=False)
        self.model.set_weights(old_weights)
        old_potential = self._compute_observables(full_x, full_y, backward=True, extra=False)['loss']
        old_grad      = self.model.copy(grad=True)
        self.model.set_weights(new_weights)

        # ΔU = U(new) − U(old)
        dU = new_potential - old_potential

        # Calcolo logM
        logM = 0.0
        for layer in self.model.diff_layers:
            og = old_grad[layer]
            ng = new_grad[layer]
            # logM layer‐wise
            logM_vector = (
                -(new_weights[layer] - old_weights[layer]) * (ng + og) / 2.0
                + (ng**2.0 - og**2.0) * (self.pars['dt']**2.0) / (8.0 * self.pars['m'])
            )
            logM += logM_vector.detach().cpu().numpy().sum()

        # Kinetica vecchia (su old_momenta)
        new_K = self._compute_K()

        self.check_bal= {
            'move'      : move,
            'logM'      : logM,
            'dU'        : dU.item() if isinstance(dU, torch.Tensor) else dU,
            'dH'        : (logM + (dU.item() if isinstance(dU, torch.Tensor) else dU)),
            'wratio'    : (logM / dU.item()) if (isinstance(dU, torch.Tensor) and dU.item() != 0)
                          else (logM / dU) if (not isinstance(dU, torch.Tensor) and dU != 0)
                          else float('inf'),
            'K'         : old_K,
            'dK'        : new_K - old_K
        }


        
    def _check_data(self, move, dt_move):
    # calcolo pesi correnti vs pesi precedenti
        btb_d = compute_d(self.model.weights, self.prev_weights).item()
        d2_val = compute_d2(self.model.weights, self.w0).item()
        q_val  = compute_q(self.model.weights, self.w0).item()
        n_moves = move + 1

        self.check_data = {
           'd2'       : d2_val,
           'btb_d'    : btb_d,
           'dt_move'  : dt_move,
           'eff_v'    : btb_d / dt_move,
           'eff_D'    : d2_val / (dt_move * n_moves) if n_moves else float('inf'),
           'similarity': q_val,
        }

    
    self.check.update(self.check_data)



    ## Initialize check detailed balance dictionary
    #


    ## Save PL sampling data or detailed balace check data
    def _save(self, dikt, filename, header=False):
        if header:
            with open(filename, 'w') as f:
                header, line = '', ''
                for key in dikt:
                    header = header + f'{key}\t'
                    line = line + f'{dikt[key]}\t'
                print(header[:-1], file=f)
                print(line[:-1], file=f)
        else:
            with open(filename, 'a') as f:
                line = ''
                for key in dikt: line = line + f'{dikt[key]}\t'
                print(line[:-1], file=f)


    ## Save PL sampling model weights, and torch generator state
    def _wsave(self, move):
        self.model.save(f'{self.weights_dir}/{self.prefixes["w"]}{move}.pt')
        # Ensure momenta is not None before saving
        if hasattr(self, 'momenta') and self.momenta is not None:
            torch.save(self.momenta, f'{self.weights_dir}/{self.prefixes["m"]}{move}.pt')
        else:
            print("Warning: Momenta is not initialized. Skipping saving momenta.")

        if move == 0:
            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)
            # Check if source file exists before copying
            if os.path.exists(f'{self.results_dir}/torch_state.npy'):
                 shutil.copy(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
            else:
                 print("Warning: torch_state.npy not found. Cannot create prev_torch_state.npy.")

        else:
            # Check if prev_torch_state.npy exists before attempting to remove
            if os.path.exists(f'{self.results_dir}/prev_torch_state.npy'):
                os.remove(f'{self.results_dir}/prev_torch_state.npy')
            else:
                print("Warning: prev_torch_state.npy not found. Cannot remove.")

            # Check if torch_state.npy exists before renaming
            if os.path.exists(f'{self.results_dir}/torch_state.npy'):
                 os.rename(f'{self.results_dir}/torch_state.npy', f'{self.results_dir}/prev_torch_state.npy')
            else:
                 print("Warning: torch_state.npy not found. Cannot rename.")

            save_torch_state(f'{self.results_dir}/torch_state.npy', self.generator)


    ## Print PL sampling data
    def _status(self):
        # Check if self.data is initialized and has 'time' key
        if not hasattr(self, 'data') or 'time' not in self.data:
             print("Warning: self.data not properly initialized for printing status.")
             return

        self.data['time_h'] = self.data["time"] / 3600.

        line = ''
        for key, _, fp in self.printing_stuff['data']:

             value_to_format = self.data.get(key, 0.0)
             line = f'{line}|{format(value_to_format, f".{fp}f"):^12}'
        line = f'{line}|' + ''.join([' ']*5)
        for key, _, fp in self.printing_stuff['check']:

             value_to_format = self.check.get(key, 0.0)
             line = f'{line}|{format(value_to_format, f".{fp}f"):^12}'
        line = f'{line}|'

        self.data.pop('time_h')
        print(f'{line}\n{self.separator}')


    ## Prepare printing of PL data
    def _prepare_printing(self):
        self.printing_stuff = {
            'data': [
                ['move',   'move',   0],
                ['loss',   'U',      5],
                ['cost',   'loss',   5],
                ['metric', 'metric', 5],
                ['mod2',   'mod2',   1],
                ['time_h', 'time', 2],
                ],

            'check': [
                ['move',      'move',    0],
                ['dK',        'dK',      3],
                ['wratio',    'logM/dU', 3],
                ['dt_move',   'dt_move', 3],
                ['dt_pl',     'dt_pl',   4],
                ['dt_adj',  'tadj',  4],
                ['dt_bal',  'tbal',  4],
                ['d2',        'd2',      3],
                ['eff_v',     'eff_v',   4],
                ['eff_D',     'eff_D',   4],
                ['similarity', 'q',      3],
            ]
        }

        # Costruisco separatori e header
        self.separator = (
            '-' * (13 * len(self.printing_stuff['data']) + 1)
            + ' ' * 5
            + '-' * (13 * len(self.printing_stuff['check']) + 1)
        )

        header = ''
        for _, symbol, _ in self.printing_stuff['data']:
            header = f'{header}|{symbol:^12}'
        header = f'{header}|' + ' ' * 5
        for _, symbol, _ in self.printing_stuff['check']:
            header = f'{header}|{symbol:^12}'
        header = f'{header}|'

        print(f'// {self.name} status register:')
        print(f'{self.separator}\n{header}\n{self.separator}')


    ## Convert to correct types each dictionary values
    def _to_types(self):
        types_and_keys = {
                'pars': [
                    (int, ['moves', 'mbs', 'max_extractions', 'extractions', 'max_adj_step', 'min_adj_step', 'opt_streak', 'seed', 'isteps']), # Added isteps to int
                    (bool, ['lob']),
                    (float, ['lamda', 'T', 'T_mb', 'dt', 'm1', 'std', 'k_mb', 'k_wn', 'threshold_est', 'threshold_adj', 'dtpl']), # Added dtpl
                ],
                'vpars': [
                    (int, ['move', 'streak', 'adj_step']),
                    (float, ['std', 'T_mb', 'k_mb', 'k_wn']),
                ],
                'data': [
                    (int, ['move']),
                    (float, ['loss', 'cost', 'mod2', 'metric', 'time']), # Added missing keys
                ],
                'check': [
                    (int, ['move']),
                    (float, ['logM', 'dU', 'dH', 'wratio', 'K', 'dK', 'btb_d', 'step_dt', 'eff_v', 'eff_D', 'similarity']), # Added missing keys
                ]
        }

        self.pars = self._correct(self.pars, types_and_keys['pars'])
        self.vpars = self._correct(self.vpars, types_and_keys['vpars']) if hasattr(self, 'vpars') else {} # Handle case where vpars might not be initialized
        self.data = self._correct(self.data, types_and_keys['data']) if hasattr(self, 'data') else {} # Handle case where data might not be initialized
        self.check = self._correct(self.check, types_and_keys['check']) if hasattr(self, 'check') else {} # Handle case where check might not be initialized

    def _correct(self, d, types_and_keys):
        corrected_d = {}
        for key in d:
            found = False
            for _type, keys in types_and_keys:
                if key in keys:
                    try:
                        corrected_d[key] = _type(d[key])
                        found = True
                        break
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert key '{key}' with value '{d[key]}' to type {_type.__name__}. Keeping original type.")
                        corrected_d[key] = d[key] # Keep original value if conversion fails
            if not found:
                corrected_d[key] = d[key]
        return corrected_d
