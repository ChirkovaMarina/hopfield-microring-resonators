"""Experiment B: associative memory recall tests."""

import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.network import NetParams, PhotonicHopfieldNetwork
from src.physics import MRRParams


class ExpB_AssocRecall:
    """Associative recall tests."""
    
    def __init__(self, n_nodes = 8, 
                 mrr_params: MRRParams = None,
                 net_params: NetParams = None):
        self.N = n_nodes
        self.mrr_params = mrr_params or MRRParams()
        self.net_params = net_params or NetParams(N=n_nodes)
        self.network = PhotonicHopfieldNetwork(n_nodes, self.mrr_params, self.net_params)
        self.results = {}
        
    def generate_random_patterns(self, n_patterns):
        """Generate random binary patterns"""
        return np.random.choice([-1, 1], size=(n_patterns, self.N))
    
    def corrupt_pattern(self, pattern, k: int):
        """Flip k random bits in pattern"""
        corrupted = pattern.copy()
        flip_indices = np.random.choice(self.N, size=k, replace=False)
        corrupted[flip_indices] *= -1
        return corrupted
    
    def run_recall_test(self, patterns, 
                        corruption_levels: list = None,
                        n_trials = 50,
                        use_optical = False):
        """Test recall accuracy for different corruption levels."""
        if corruption_levels is None:
            corruption_levels = list(range(0, self.N//2 + 1))
        
        # Store patterns
        self.network.store_patterns(patterns)
        
        results = {
            'corruption_levels': corruption_levels,
            'success_rates': [],
            'convergence_times': [],
            'overlap_histories': []
        }
        
        for k in tqdm(corruption_levels, desc="Testing corruption levels"):
            successes = 0
            conv_times = []
            
            for trial in range(n_trials):
                # Pick random stored pattern
                p_idx = np.random.randint(len(patterns))
                target = patterns[p_idx]
                
                # Corrupt it
                initial = self.corrupt_pattern(target, k)
                
                # Recall
                if use_optical:
                    result = self.network.simulate_optical(initial)
                else:
                    result = self.network.recall_discrete(initial)
                
                # Check success
                if np.array_equal(result['final_state'], target):
                    successes += 1
                conv_times.append(result.get('iterations', len(result.get('t', [0]))))
            
            results['success_rates'].append(successes / n_trials)
            results['convergence_times'].append(np.mean(conv_times))
        
        self.results['recall_test'] = results
        return results
    
    def run_capacity_test(self, max_patterns = None,
                          n_trials = 20):
        """Test storage capacity of the network."""
        if max_patterns is None:
            max_patterns = int(0.15 * self.N)  # Theoretical limit ~0.138N
        
        pattern_counts = list(range(1, max_patterns + 1))
        success_rates = []
        
        for P in tqdm(pattern_counts, desc="Testing capacity"):
            successes = 0
            total = 0
            
            for _ in range(n_trials):
                # Generate and store patterns
                patterns = self.generate_random_patterns(P)
                self.network.store_patterns(patterns)
                
                # Test recall of each pattern
                for p_idx, pattern in enumerate(patterns):
                    result = self.network.recall_discrete(pattern.copy())
                    if np.array_equal(result['final_state'], pattern):
                        successes += 1
                    total += 1
            
            success_rates.append(successes / total)
        
        self.results['capacity'] = {
            'pattern_counts': pattern_counts,
            'success_rates': success_rates,
            'theoretical_limit': 0.138 * self.N
        }
        return self.results['capacity']
    
    def run_basin_analysis(self, pattern,
                           n_samples: int = 1000):
        """Analyze basin of attraction for a pattern."""
        self.network.store_patterns(pattern.reshape(1, -1))
        
        # Test from all Hamming distances
        distances = list(range(self.N + 1))
        convergence_probs = []
        
        for d in distances:
            converged = 0
            n_configs = min(n_samples, math.comb(self.N, d))
            
            for _ in range(n_configs):
                # Create state at Hamming distance d
                initial = pattern.copy()
                flip_idx = np.random.choice(self.N, size=d, replace=False)
                initial[flip_idx] *= -1
                
                result = self.network.recall_discrete(initial)
                if np.array_equal(result['final_state'], pattern):
                    converged += 1
            
            convergence_probs.append(converged / n_configs if n_configs > 0 else 0)
        
        self.results['basin'] = {
            'distances': distances,
            'convergence_probs': convergence_probs,
            'pattern': pattern
        }
        return self.results['basin']
    
    def plot_results(self, save_path = None):
        """Plot results."""
        n_plots = sum(k in self.results for k in ['recall_test', 'capacity', 'basin'])
        fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(5*max(n_plots, 1), 4))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Recall accuracy
        if 'recall_test' in self.results:
            ax = axes[plot_idx]
            data = self.results['recall_test']
            ax.plot(data['corruption_levels'], data['success_rates'], 
                   'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Number of Corrupted Bits')
            ax.set_ylabel('Recall Success Rate')
            ax.set_title('Pattern Recall vs Corruption')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Capacity
        if 'capacity' in self.results:
            ax = axes[plot_idx]
            data = self.results['capacity']
            ax.plot(data['pattern_counts'], data['success_rates'],
                   'go-', linewidth=2, markersize=8)
            ax.axvline(x=data['theoretical_limit'], color='r', 
                      linestyle='--', label=f'Theory: {data["theoretical_limit"]:.1f}')
            ax.set_xlabel('Number of Stored Patterns')
            ax.set_ylabel('Perfect Recall Rate')
            ax.set_title(f'Storage Capacity (N={self.N})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Basin of attraction
        if 'basin' in self.results:
            ax = axes[plot_idx]
            data = self.results['basin']
            ax.bar(data['distances'], data['convergence_probs'],
                  color='purple', alpha=0.7)
            ax.set_xlabel('Hamming Distance from Pattern')
            ax.set_ylabel('Convergence Probability')
            ax.set_title('Basin of Attraction')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
