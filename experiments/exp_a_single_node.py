"""Experiment A: single microring resonator characterization."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hopfield_nn.physics import MRRParams, MicroringResonator


class ExpA_SingleNode:
    """Single-node tests."""
    
    def __init__(self, mrr_params: MRRParams = None):
        self.params = mrr_params or MRRParams()
        self.mrr = MicroringResonator(self.params)
        self.results = {}
        
    def run_hysteresis(self, P_min = 1e-6, P_max = 10e-3,
                       n_points = 200, detuning = 0.0):
        """Characterize hysteresis loop"""
        P_range = np.linspace(P_min, P_max, n_points)
        P_out_up, P_out_down = self.mrr.hysteresis_sweep(P_range, detuning)
        
        self.results['hysteresis'] = {
            'P_in': P_range,
            'P_out_up': P_out_up,
            'P_out_down': P_out_down,
            'detuning': detuning
        }
        return self.results['hysteresis']
    
    def run_switching(self, P_low = 1e-3, P_high = 5e-3,
                      t_switch = 1e-9, t_total = 10e-9):
        """Measure switching dynamics"""
        self.mrr.reset_state()
        
        # Initialize in low state
        self.mrr.steady_state(P_low)
        
        # Define input with step
        def s_in(t):
            P = P_high if t > t_total/4 else P_low
            return np.sqrt(P)
        
        # Simulate
        result = self.mrr.simulate((0, t_total), s_in, dt=1e-12)
        
        # Find switching time (10%-90%)
        I_norm = result['intensity'] / np.max(result['intensity'])
        try:
            t_10 = result['t'][np.where(I_norm > 0.1)[0][0]]
            t_90 = result['t'][np.where(I_norm > 0.9)[0][0]]
            t_rise = t_90 - t_10
        except:
            t_rise = np.nan
        
        self.results['switching'] = {
            't': result['t'],
            'intensity': result['intensity'],
            'P_low': P_low,
            'P_high': P_high,
            't_rise': t_rise
        }
        return self.results['switching']
    
    def run_noise_stability(self, P_operating = 3e-3,
                            noise_levels = None,
                            n_trials = 100,
                            t_trial = 10e-9):
        """Test stability under noise"""
        if noise_levels is None:
            noise_levels = np.logspace(-3, -1, 10)
        
        flip_rates = []
        
        for noise in tqdm(noise_levels, desc="Noise sweep"):
            flips = 0
            for _ in range(n_trials):
                self.mrr.reset_state()
                # Initialize in high state
                self.mrr.steady_state(P_operating)
                initial_state = 1 if np.abs(self.mrr.a)**2 > 1e-6 else -1
                
                # Add noise to input
                def s_in(t):
                    return np.sqrt(P_operating) * (1 + noise * np.random.randn())
                
                # Simulate
                result = self.mrr.simulate((0, t_trial), s_in, dt=1e-12)
                
                # Check if flipped
                final_intensity = result['intensity'][-1]
                final_state = 1 if final_intensity > 1e-6 else -1
                if final_state != initial_state:
                    flips += 1
            
            flip_rates.append(flips / n_trials)
        
        self.results['noise_stability'] = {
            'noise_levels': noise_levels,
            'flip_rates': np.array(flip_rates)
        }
        return self.results['noise_stability']
    
    def plot_results(self, save_path = None):
        """Plot results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Hysteresis
        if 'hysteresis' in self.results:
            ax = axes[0, 0]
            data = self.results['hysteresis']
            ax.plot(data['P_in']*1e3, data['P_out_up']*1e3, 'b-', 
                   label='Up sweep', linewidth=2)
            ax.plot(data['P_in']*1e3, data['P_out_down']*1e3, 'r--', 
                   label='Down sweep', linewidth=2)
            ax.set_xlabel('Input Power (mW)')
            ax.set_ylabel('Output Power (mW)')
            ax.set_title('Optical Bistability Hysteresis')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Switching dynamics
        if 'switching' in self.results:
            ax = axes[0, 1]
            data = self.results['switching']
            ax.plot(data['t']*1e9, data['intensity']*1e6, 'b-', linewidth=2)
            ax.axvline(x=data['t'][len(data['t'])//4]*1e9, color='r', 
                      linestyle='--', label='Input step')
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Cavity Intensity (a.u.)')
            ax.set_title(f'Switching Dynamics (t_rise = {data["t_rise"]*1e9:.2f} ns)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Noise stability
        if 'noise_stability' in self.results:
            ax = axes[1, 0]
            data = self.results['noise_stability']
            ax.semilogx(data['noise_levels']*100, data['flip_rates'], 
                       'bo-', linewidth=2, markersize=8)
            ax.set_xlabel('Noise Level (%)')
            ax.set_ylabel('Flip Probability')
            ax.set_title('State Stability vs Input Noise')
            ax.grid(True, alpha=0.3)
        
        # Parameter summary
        ax = axes[1, 1]
        ax.axis('off')
        params_text = f"""MRR Parameters:"""
        ax.text(0.1, 0.5, params_text, fontsize=12, family='monospace',
               verticalalignment='center')
        ax.set_title('Device Parameters')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
