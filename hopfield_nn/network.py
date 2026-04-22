"""Photonic Hopfield network model."""

from dataclasses import dataclass

import numpy as np

from .physics import MRRParams, MicroringResonator


@dataclass
class NetParams:
    """Basic network parameters."""
    N: int = 4                    # Number of nodes
    alpha: float = 0.5            # Coupling strength
    tau_delay: float = 0.0        # Interconnect delay [s]
    loss_coupling: float = 0.1    # Coupling loss fraction
    asymmetry: float = 0.0        # Weight matrix asymmetry factor
    
    # Noise parameters
    amplitude_noise: float = 0.0  # Relative amplitude noise
    phase_noise: float = 0.0      # Phase noise [rad]
    
    # Simulation parameters
    t_sim: float = 50e-9          # Total simulation time [s]
    dt: float = 1e-12             # Time step [s]


class PhotonicHopfieldNetwork:
    """Hopfield network built from MRR-like nodes."""
    
    def __init__(self, n_nodes = 4, 
                 mrr_params: MRRParams = None,
                 net_params: NetParams = None):
        """Initialize network."""
        self.N = n_nodes
        self.mrr_params = mrr_params or MRRParams()
        self.net_params = net_params or NetParams(N=n_nodes)
        
        # Create MRR nodes
        self.nodes = [MicroringResonator(self.mrr_params) for _ in range(n_nodes)]
        
        # weight matrix
        self.W = np.zeros((n_nodes, n_nodes))
        
        # saved patterns
        self.patterns = []
        
        # history
        self.history = []
        
    def store_patterns(self, patterns):
        """Store patterns using Hebbian learning."""
        patterns = np.array(patterns)
        if patterns.ndim == 1:
            patterns = patterns.reshape(1, -1)
            
        P, N = patterns.shape
        assert N == self.N, f"Pattern size {N} doesn't match network size {self.N}"
        
        self.patterns = patterns.copy()
        
        # Hebbian learning rule
        self.W = np.zeros((N, N))
        for p in range(P):
            xi = patterns[p]
            self.W += np.outer(xi, xi)
        self.W /= N
        
        # no self links
        np.fill_diagonal(self.W, 0)
        
        # Add asymmetry if specified
        if self.net_params.asymmetry > 0:
            A = np.random.randn(N, N) * self.net_params.asymmetry
            A = (A - A.T) / 2  # Anti-symmetric perturbation
            self.W += A
    
    def compute_energy(self, state):
        """Compute Hopfield energy function."""
        return -0.5 * np.dot(state, np.dot(self.W, state))
    
    def local_field(self, state, i: int):
        """Compute local field at node i."""
        return np.dot(self.W[i], state)
    
    def update_discrete(self, state, 
                        mode: str = 'async',
                        noise = 0.0):
        """Perform one update step (discrete Hopfield)."""
        new_state = state.copy()
        
        if mode == 'async':
            # random update order
            order = np.random.permutation(self.N)
            for i in order:
                h = self.local_field(new_state, i)
                if noise > 0:
                    # Stochastic update
                    prob = 1 / (1 + np.exp(-2 * h / noise))
                    new_state[i] = 1 if np.random.random() < prob else -1
                else:
                    # usual update
                    new_state[i] = 1 if h >= 0 else -1
        else:
            # sync update
            h = np.dot(self.W, state)
            if noise > 0:
                prob = 1 / (1 + np.exp(-2 * h / noise))
                new_state = np.where(np.random.random(self.N) < prob, 1, -1)
            else:
                new_state = np.sign(h)
                new_state[new_state == 0] = 1
                
        return new_state.astype(int)
    
    def recall_discrete(self, initial_state,
                        max_iter: int = 100,
                        noise = 0.0):
        """Perform pattern recall using discrete dynamics."""
        state = initial_state.copy()
        history = [state.copy()]
        energies = [self.compute_energy(state)]
        
        for i in range(max_iter):
            new_state = self.update_discrete(state, 'async', noise)
            history.append(new_state.copy())
            energies.append(self.compute_energy(new_state))
            
            # stop if converged
            if np.array_equal(new_state, state):
                break
            state = new_state
        
        # Find closest stored pattern
        if len(self.patterns) > 0:
            overlaps = [np.mean(state == p) for p in self.patterns]
            best_match = np.argmax(overlaps)
            success = overlaps[best_match] == 1.0
        else:
            best_match = -1
            success = False
            overlaps = []
        
        return {
            'final_state': state,
            'history': np.array(history),
            'energies': np.array(energies),
            'iterations': len(history) - 1,
            'converged': len(history) < max_iter + 1,
            'best_match': best_match,
            'overlaps': overlaps,
            'success': success
        }
    
    def state_to_optical(self, state, P_base: float = 1e-3):
        """Convert binary state to optical input powers."""
        # Map -1 -> low power, +1 -> high power
        return P_base * (1 + 0.5 * state)
    
    def optical_to_state(self, intensities, 
                         threshold: float = None):
        """Convert optical intensities to binary state."""
        if threshold is None:
            threshold = np.median(intensities)
        return np.where(intensities > threshold, 1, -1)
    
    def simulate_optical(self, initial_state,
                         t_total: float = None,
                         dt = None,
                         include_thermal = False):
        """Simulate network with full optical dynamics."""
        if t_total is None:
            t_total = self.net_params.t_sim
        if dt is None:
            dt = self.net_params.dt
            
        n_steps = int(t_total / dt)
        t = np.arange(n_steps) * dt
        
        # Initialize node states
        P_base = 1e-3  # Base power 1 mW
        for i, node in enumerate(self.nodes):
            node.reset_state()
            # Set initial state through input power
            P_init = P_base * (2 if initial_state[i] == 1 else 0.5)
            result = node.steady_state(P_init)
            node.a = result['a']
            node.N_fc = result['N_fc']
        
        # Storage for time series
        intensities = np.zeros((self.N, n_steps))
        states = np.zeros((self.N, n_steps), dtype=int)
        carriers = np.zeros((self.N, n_steps))
        
        # Get initial intensities for threshold
        init_intensities = np.array([np.abs(node.a)**2 for node in self.nodes])
        threshold = np.median(init_intensities) * 1.5
        
        # Time evolution
        for step in range(n_steps):
            # Current states and intensities
            current_intensities = np.array([np.abs(node.a)**2 for node in self.nodes])
            current_states = self.optical_to_state(current_intensities, threshold)
            
            # Store
            intensities[:, step] = current_intensities
            states[:, step] = current_states
            carriers[:, step] = np.array([node.N_fc for node in self.nodes])
            
            # Compute inputs for next step
            for i, node in enumerate(self.nodes):
                # Base input
                P_in = P_base
                
                # Add contributions from other nodes
                for j in range(self.N):
                    if i != j:
                        # Coupling with loss and possible delay
                        coupling = self.net_params.alpha * self.W[i, j]
                        coupling *= (1 - self.net_params.loss_coupling)
                        
                        # Add noise
                        if self.net_params.amplitude_noise > 0:
                            coupling *= (1 + self.net_params.amplitude_noise * 
                                       np.random.randn())
                        
                        # Contribution from node j
                        P_in += coupling * current_intensities[j]
                
                # Ensure positive
                P_in = max(P_in, 1e-6)
                
                # Single time step update
                s_in = np.sqrt(P_in)
                if self.net_params.phase_noise > 0:
                    s_in *= np.exp(1j * self.net_params.phase_noise * np.random.randn())
                
                # Euler step for speed (could use RK4 for accuracy)
                y = np.array([node.a.real, node.a.imag, node.N_fc, node.T])
                dy = node.equations(t[step], y, s_in, include_thermal)
                y_new = y + dt * dy
                
                node.a = y_new[0] + 1j * y_new[1]
                node.N_fc = max(y_new[2], 0)
                node.T = y_new[3]
        
        # Final state
        final_intensities = intensities[:, -1]
        final_state = self.optical_to_state(final_intensities, threshold)
        
        # Analyze convergence
        if len(self.patterns) > 0:
            overlaps = [np.mean(final_state == p) for p in self.patterns]
            best_match = np.argmax(overlaps)
            success = overlaps[best_match] == 1.0
        else:
            best_match = -1
            success = False
            overlaps = []
        
        return {
            't': t,
            'intensities': intensities,
            'states': states,
            'carriers': carriers,
            'final_state': final_state,
            'best_match': best_match,
            'overlaps': overlaps,
            'success': success
        }
