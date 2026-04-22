"""Experiment C: physical parameter sweeps."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hopfield_nn.network import NetParams, PhotonicHopfieldNetwork
from hopfield_nn.physics import MRRParams


class ExpC_Physics:
    """Parameter sweeps."""
    
    def __init__(self, n_nodes = 8):
        self.N = n_nodes
        self.results = {}
        
    def parameter_sweep(self, param_name, 
                        param_values: np.ndarray,
                        n_patterns = 2,
                        n_trials = 30):
        """Sweep a physical parameter and measure network performance."""
        success_rates = []
        convergence_times = []
        
        for val in tqdm(param_values, desc=f"Sweeping {param_name}"):
            # Create network with modified parameter
            mrr_params = MRRParams()
            net_params = NetParams(N=self.N)
            
            # Set parameter
            if hasattr(mrr_params, param_name):
                setattr(mrr_params, param_name, val)
                mrr_params.__post_init__()  # Recalculate derived params
            elif hasattr(net_params, param_name):
                setattr(net_params, param_name, val)
            
            network = PhotonicHopfieldNetwork(self.N, mrr_params, net_params)
            
            # Generate patterns
            patterns = np.random.choice([-1, 1], size=(n_patterns, self.N))
            network.store_patterns(patterns)
            
            # Test recall
            successes = 0
            times = []
            
            for _ in range(n_trials):
                p_idx = np.random.randint(n_patterns)
                # Corrupt 1-2 bits
                initial = patterns[p_idx].copy()
                k = np.random.randint(1, 3)
                flip_idx = np.random.choice(self.N, size=k, replace=False)
                initial[flip_idx] *= -1
                
                result = network.recall_discrete(initial, noise = 0.2)
                if np.array_equal(result['final_state'], patterns[p_idx]):
                    successes += 1
                times.append(result['iterations'])
            
            success_rates.append(successes / n_trials)
            convergence_times.append(np.mean(times))
        
        self.results[param_name] = {
            'values': param_values,
            'success_rates': np.array(success_rates),
            'convergence_times': np.array(convergence_times)
        }
        return self.results[param_name]
    
    def run_q_factor_sweep(self, Q_range = None):
        """Sweep Q factor"""
        if Q_range is None:
            Q_range = np.logspace(3, 5, 15)
        return self.parameter_sweep('Q_int', Q_range)
    
    def run_noise_sweep(self, noise_range = None):
        """Sweep amplitude noise"""
        if noise_range is None:
            noise_range = np.linspace(0, 0.3, 15)
        return self.parameter_sweep('amplitude_noise', noise_range)
    
    def run_asymmetry_sweep(self, asym_range = None):
        """Sweep weight matrix asymmetry"""
        if asym_range is None:
            asym_range = np.linspace(0, 0.5, 15)
        return self.parameter_sweep('asymmetry', asym_range)
    
    def run_coupling_loss_sweep(self, loss_range = None):
        """Sweep coupling loss"""
        if loss_range is None:
            loss_range = np.linspace(0, 0.5, 15)
        return self.parameter_sweep('loss_coupling', loss_range)
    
    def create_stability_map(self, param1, param1_range: np.ndarray,
                             param2, param2_range: np.ndarray,
                             n_trials = 20):
        """Create 2D stability map for two parameters."""
        success_map = np.zeros((len(param1_range), len(param2_range)))
        
        for i, v1 in enumerate(tqdm(param1_range, desc="Creating stability map")):
            for j, v2 in enumerate(param2_range):
                mrr_params = MRRParams()
                net_params = NetParams(N=self.N)
                
                # Set parameters
                for p, v in [(param1, v1), (param2, v2)]:
                    if hasattr(mrr_params, p):
                        setattr(mrr_params, p, v)
                    elif hasattr(net_params, p):
                        setattr(net_params, p, v)
                
                if hasattr(mrr_params, param1) or hasattr(mrr_params, param2):
                    mrr_params.__post_init__()
                
                network = PhotonicHopfieldNetwork(self.N, mrr_params, net_params)
                
                # Test
                patterns = np.random.choice([-1, 1], size=(2, self.N))
                network.store_patterns(patterns)
                
                successes = 0
                for _ in range(n_trials):
                    p_idx = np.random.randint(2)
                    initial = patterns[p_idx].copy()
                    initial[np.random.randint(self.N)] *= -1
                    result = network.recall_discrete(initial)
                    if np.array_equal(result['final_state'], patterns[p_idx]):
                        successes += 1
                
                success_map[i, j] = successes / n_trials
        
        self.results['stability_map'] = {
            'param1': param1,
            'param2': param2,
            'param1_range': param1_range,
            'param2_range': param2_range,
            'success_map': success_map
        }
        return self.results['stability_map']
    
    def plot_results(self, save_path = None):
        """Plot results."""
        # Count 1D sweeps
        sweep_keys = [k for k in self.results.keys() if k != 'stability_map']
        n_sweeps = len(sweep_keys)
        has_map = 'stability_map' in self.results
        
        if has_map:
            fig = plt.figure(figsize=(14, 5))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
            axes = [fig.add_subplot(gs[0, i]) for i in range(min(n_sweeps, 2))]
            ax_map = fig.add_subplot(gs[0, 2])
        else:
            n_cols = min(n_sweeps, 3)
            n_rows = (n_sweeps + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_sweeps == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
        
        # Plot 1D sweeps
        colors = plt.cm.tab10.colors
        for idx, key in enumerate(sweep_keys[:len(axes)]):
            ax = axes[idx]
            data = self.results[key]
            ax.plot(data['values'], data['success_rates'],
                   'o-', color=colors[idx], linewidth=2, markersize=6)
            ax.set_xlabel(key.replace('_', ' ').title())
            ax.set_ylabel('Recall Success Rate')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Effect of {key}')
        
        # Plot stability map
        if has_map:
            data = self.results['stability_map']
            im = ax_map.imshow(data['success_map'], origin='lower', aspect='auto',
                              extent=[data['param2_range'][0], data['param2_range'][-1],
                                     data['param1_range'][0], data['param1_range'][-1]],
                              cmap='RdYlGn', vmin=0, vmax=1)
            ax_map.set_xlabel(data['param2'].replace('_', ' ').title())
            ax_map.set_ylabel(data['param1'].replace('_', ' ').title())
            ax_map.set_title('Stability Region')
            plt.colorbar(im, ax=ax_map, label='Success Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
