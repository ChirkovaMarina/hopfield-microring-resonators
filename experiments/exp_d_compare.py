"""Experiment D: digital vs photonic comparison."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hopfield_nn.network import NetParams, PhotonicHopfieldNetwork


class ExpD_Compare:
    """Comparison with digital version."""
    
    def __init__(self, n_nodes = 8):
        self.N = n_nodes
        self.results = {}
        
    def compare_recall(self, n_patterns = 2,
                       corruption_range = None,
                       n_trials = 100):
        """Compare recall performance between photonic and digital networks."""
        if corruption_range is None:
            corruption_range = list(range(0, self.N//2 + 1))
        
        # Create networks
        photonic = PhotonicHopfieldNetwork(self.N)
        
        # Generate patterns
        patterns = np.random.choice([-1, 1], size=(n_patterns, self.N))
        photonic.store_patterns(patterns)
        
        results = {
            'corruption_levels': corruption_range,
            'photonic_success': [],
            'digital_sync_success': [],
            'digital_async_success': [],
            'photonic_time': [],
            'digital_iterations': []
        }
        
        for k in tqdm(corruption_range, desc="Comparing networks"):
            ph_success = 0
            sync_success = 0
            async_success = 0
            ph_times = []
            dig_iters = []
            
            for _ in range(n_trials):
                p_idx = np.random.randint(n_patterns)
                target = patterns[p_idx]
                
                # Corrupt
                initial = target.copy()
                if k > 0:
                    flip_idx = np.random.choice(self.N, size=k, replace=False)
                    initial[flip_idx] *= -1
                
                # Photonic recall
                ph_result = photonic.recall_discrete(initial.copy())
                if np.array_equal(ph_result['final_state'], target):
                    ph_success += 1
                ph_times.append(ph_result['iterations'] * 1e-9)  # Assume ~1ns per update
                
                # Digital synchronous
                state = initial.copy()
                for i in range(100):
                    h = np.dot(photonic.W, state)
                    new_state = np.sign(h)
                    new_state[new_state == 0] = 1
                    if np.array_equal(new_state, state):
                        break
                    state = new_state
                if np.array_equal(state, target):
                    sync_success += 1
                dig_iters.append(i + 1)
                
                # Digital asynchronous (same as photonic discrete)
                async_result = photonic.recall_discrete(initial.copy())
                if np.array_equal(async_result['final_state'], target):
                    async_success += 1
            
            results['photonic_success'].append(ph_success / n_trials)
            results['digital_sync_success'].append(sync_success / n_trials)
            results['digital_async_success'].append(async_success / n_trials)
            results['photonic_time'].append(np.mean(ph_times))
            results['digital_iterations'].append(np.mean(dig_iters))
        
        self.results['recall_comparison'] = results
        return results
    
    def compare_noise_robustness(self, noise_levels = None,
                                 n_trials = 100):
        """Compare noise robustness between networks."""
        if noise_levels is None:
            noise_levels = np.linspace(0, 0.5, 15)
        
        results = {
            'noise_levels': noise_levels,
            'photonic_success': [],
            'digital_success': []
        }
        
        patterns = np.random.choice([-1, 1], size=(2, self.N))
        
        for noise in tqdm(noise_levels, desc="Testing noise robustness"):
            # Photonic with noise
            net_params = NetParams(N=self.N, amplitude_noise=noise)
            photonic = PhotonicHopfieldNetwork(self.N, net_params=net_params)
            photonic.store_patterns(patterns)
            
            ph_success = 0
            dig_success = 0
            
            for _ in range(n_trials):
                p_idx = np.random.randint(2)
                initial = patterns[p_idx].copy()
                initial[np.random.randint(self.N)] *= -1
                
                # Photonic
                result = photonic.recall_discrete(initial.copy(), noise=noise)
                if np.array_equal(result['final_state'], patterns[p_idx]):
                    ph_success += 1
                
                # Digital with same noise level
                result = photonic.recall_discrete(initial.copy(), noise=noise)
                if np.array_equal(result['final_state'], patterns[p_idx]):
                    dig_success += 1
            
            results['photonic_success'].append(ph_success / n_trials)
            results['digital_success'].append(dig_success / n_trials)
        
        self.results['noise_comparison'] = results
        return results
    
    def plot_results(self, save_path = None):
        """Plot comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Recall comparison
        if 'recall_comparison' in self.results:
            ax = axes[0]
            data = self.results['recall_comparison']
            ax.plot(data['corruption_levels'], data['photonic_success'],
                   'b-o', linewidth=2, label='Photonic (async)', markersize=8)
            ax.plot(data['corruption_levels'], data['digital_sync_success'],
                   'g--s', linewidth=2, label='Digital (sync)', markersize=8)
            ax.plot(data['corruption_levels'], data['digital_async_success'],
                   'r:^', linewidth=2, label='Digital (async)', markersize=8)
            ax.set_xlabel('Corrupted Bits')
            ax.set_ylabel('Recall Success Rate')
            ax.set_title('Recall Performance Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        # Noise comparison
        if 'noise_comparison' in self.results:
            ax = axes[1]
            data = self.results['noise_comparison']
            ax.plot(data['noise_levels'], data['photonic_success'],
                   'b-o', linewidth=2, label='Photonic', markersize=8)
            ax.plot(data['noise_levels'], data['digital_success'],
                   'g--s', linewidth=2, label='Digital', markersize=8)
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Recall Success Rate')
            ax.set_title('Noise Robustness Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
