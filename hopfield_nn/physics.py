"""Physical constants and microring resonator model."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class PhysConst:
    """Physical constants."""
    c: float = 3e8           # Speed of light [m/s]
    h: float = 6.626e-34     # Planck constant [J*s]
    hbar: float = 1.055e-34  # Reduced Planck constant [J*s]
    e: float = 1.602e-19     # Elementary charge [C]
    k_B: float = 1.381e-23   # Boltzmann constant [J/K]
    n_Si: float = 3.48       # Refractive index of silicon at 1550nm


@dataclass
class MRRParams:
    """Basic microring parameters."""
    # geometry
    radius: float = 5e-6          # Ring radius [m]
    width: float = 450e-9         # Waveguide width [m]
    height: float = 220e-9        # Waveguide height [m]
    
    # optics
    wavelength: float = 1550e-9   # Operating wavelength [m]
    n_eff: float = 2.4            # Effective index
    n_g: float = 4.2              # Group index
    
    # Q and losses
    Q_int: float = 50000          # Intrinsic Q factor
    Q_ext: float = 20000          # External (coupling) Q factor
    alpha_prop: float = 2.0       # Propagation loss [dB/cm]
    
    # nonlinearity
    dn_dN: float = -1.73e-27      # Refractive index change per carrier [m^3]
    da_dN: float = 1.45e-21       # Absorption change per carrier [m^2]
    beta_TPA: float = 0.8e-11     # Two-photon absorption coefficient [m/W]
    
    # carrier dynamics
    tau_fc: float = 0.5e-9        # Free carrier lifetime [s]
    V_eff: float = 1e-18          # Effective mode volume [m^3]
    
    # thermal stuff
    dn_dT: float = 1.86e-4        # Thermo-optic coefficient [1/K]
    tau_th: float = 1e-6          # Thermal time constant [s]
    R_th: float = 1e4             # Thermal resistance [K/W]
    
    # detuning
    delta_0: float = 0.0          # Initial wavelength detuning [m]
    
    def __post_init__(self):
        """Update derived values."""
        self.omega_0 = 2 * np.pi * PhysConst.c / self.wavelength
        self.L = 2 * np.pi * self.radius  # Ring circumference
        self.FSR = PhysConst.c / (self.n_g * self.L)  # Free spectral range
        
        # Total Q and photon lifetime
        self.Q_tot = 1 / (1/self.Q_int + 1/self.Q_ext)
        self.tau_ph = self.Q_tot / self.omega_0
        
        # Decay rates
        self.gamma_int = self.omega_0 / (2 * self.Q_int)  # Internal loss rate
        self.gamma_ext = self.omega_0 / (2 * self.Q_ext)  # Coupling rate
        self.gamma_tot = self.gamma_int + self.gamma_ext  # Total decay rate
        
        # Resonance linewidth
        self.delta_omega = self.omega_0 / self.Q_tot


class MicroringResonator:
    """Single MRR model."""
    
    def __init__(self, params: MRRParams = None):
        self.params = params or MRRParams()
        self.reset_state()
        
    def reset_state(self):
        """Reset state."""
        self.a = 0.0 + 0.0j     # Complex field amplitude
        self.N_fc = 0.0          # Free carrier density [1/m^3]
        self.T = 300.0           # Temperature [K]
        self.state_history = []
        
    def resonance_shift(self, N_fc, dT: float = 0.0):
        """Calculate resonance frequency shift due to carriers and temperature."""
        p = self.params
        
        # Carrier-induced shift (plasma dispersion effect)
        dn_carrier = p.dn_dN * N_fc
        
        # Thermal shift
        dn_thermal = p.dn_dT * dT
        
        # Total index change
        dn_total = dn_carrier + dn_thermal
        
        # Frequency shift: delta_omega = -omega_0 * dn/n
        delta_omega = -p.omega_0 * dn_total / p.n_eff
        
        return delta_omega
    
    def carrier_absorption(self, N_fc):
        """Calculate additional absorption due to free carriers."""
        p = self.params
        # Free carrier absorption coefficient
        alpha_fc = p.da_dN * N_fc
        # Convert to decay rate
        gamma_fc = PhysConst.c * alpha_fc / (2 * p.n_g)
        return gamma_fc
    
    def carrier_generation_rate(self, intensity):
        """Calculate free carrier generation rate from TPA."""
        p = self.params
        # Two-photon absorption generates carriers
        # G = beta_TPA * I^2 / (2 * hbar * omega)
        G = p.beta_TPA * intensity**2 / (2 * PhysConst.hbar * p.omega_0)
        return G
    
    def equations(self, t, y: np.ndarray, 
                  s_in: complex, include_thermal = False):
        """Coupled differential equations for MRR dynamics."""
        p = self.params
        
        # Unpack state
        a = y[0] + 1j * y[1]
        N_fc = max(y[2], 0)  # Ensure non-negative
        T = y[3] if include_thermal else 300.0
        dT = T - 300.0
        
        # Calculate shifts and losses
        delta_omega = self.resonance_shift(N_fc, dT)
        gamma_fc = self.carrier_absorption(N_fc)
        
        # detuning from input laser
        detuning = p.delta_0 * p.omega_0 / p.wavelength + delta_omega
        
        # Total decay rate including FCA
        gamma_total = p.gamma_tot + gamma_fc
        
        # main field equation
        # da/dt = (j*detuning - gamma_total)*a + sqrt(2*gamma_ext)*s_in
        da_dt = (1j * detuning - gamma_total) * a + np.sqrt(2 * p.gamma_ext) * s_in
        
        # intensity in ring
        # |a|^2 has units of energy; convert to intensity
        energy = np.abs(a)**2
        intensity = energy * PhysConst.c / (p.n_g * p.V_eff)
        
        # carrier dynamics
        G = self.carrier_generation_rate(intensity)
        dN_dt = G - N_fc / p.tau_fc
        
        # thermal part
        if include_thermal:
            # Absorbed power heats the ring
            P_abs = 2 * gamma_fc * energy
            dT_dt = (P_abs * p.R_th - dT) / p.tau_th
        else:
            dT_dt = 0.0
        
        return np.array([da_dt.real, da_dt.imag, dN_dt, dT_dt])
    
    def simulate(self, t_span, 
                 s_in_func,
                 dt: float = 1e-12,
                 include_thermal = False):
        """Simulate MRR dynamics over time."""
        # Initial state
        y0 = np.array([self.a.real, self.a.imag, self.N_fc, self.T])
        
        # Time points
        t_eval = np.arange(t_span[0], t_span[1], dt)
        
        # Solve ODE
        def rhs(t, y):
            s_in = s_in_func(t)
            return self.equations(t, y, s_in, include_thermal)
        
        sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, 
                        method='RK45', max_step=dt)
        
        # Extract results
        a_t = sol.y[0] + 1j * sol.y[1]
        
        # Output field: s_out = s_in - sqrt(2*gamma_ext)*a
        s_in_t = np.array([s_in_func(t) for t in sol.t])
        s_out_t = s_in_t - np.sqrt(2 * self.params.gamma_ext) * a_t
        
        # Update internal state
        self.a = a_t[-1]
        self.N_fc = sol.y[2, -1]
        self.T = sol.y[3, -1]
        
        return {
            't': sol.t,
            'a': a_t,
            's_in': s_in_t,
            's_out': s_out_t,
            'N_fc': sol.y[2],
            'T': sol.y[3],
            'transmission': np.abs(s_out_t)**2 / (np.abs(s_in_t)**2 + 1e-30),
            'intensity': np.abs(a_t)**2
        }
    
    def steady_state(self, P_in, detuning: float = 0.0,
                     max_iter = 1000, tol = 1e-10):
        """Find steady-state solution for given input power."""
        p = self.params
        s_in = np.sqrt(P_in)
        
        # Initial guess
        a = s_in / p.gamma_tot
        N_fc = 0.0
        
        for _ in range(max_iter):
            a_old = a
            
            # Calculate shifts
            delta_omega = self.resonance_shift(N_fc, 0)
            gamma_fc = self.carrier_absorption(N_fc)
            total_detuning = detuning * p.omega_0 / p.wavelength + delta_omega
            
            # Steady-state field
            gamma_total = p.gamma_tot + gamma_fc
            a = np.sqrt(2 * p.gamma_ext) * s_in / (gamma_total - 1j * total_detuning)
            
            # Steady-state carriers
            energy = np.abs(a)**2
            intensity = energy * PhysConst.c / (p.n_g * p.V_eff)
            G = self.carrier_generation_rate(intensity)
            N_fc = G * p.tau_fc
            
            # stop if converged
            if np.abs(a - a_old) < tol:
                break
        
        # Output
        s_out = s_in - np.sqrt(2 * p.gamma_ext) * a
        
        return {
            'a': a,
            'N_fc': N_fc,
            'intensity': np.abs(a)**2,
            'transmission': np.abs(s_out)**2 / P_in,
            'P_out': np.abs(s_out)**2
        }
    
    def hysteresis_sweep(self, P_range, 
                         detuning: float = 0.0):
        """Perform power sweep to characterize hysteresis."""
        P_out_up = []
        P_out_down = []
        
        # Up sweep
        self.reset_state()
        for P in P_range:
            result = self.steady_state(P, detuning)
            P_out_up.append(result['P_out'])
            self.a = result['a']
            self.N_fc = result['N_fc']
        
        # Down sweep
        for P in P_range[::-1]:
            result = self.steady_state(P, detuning)
            P_out_down.append(result['P_out'])
            self.a = result['a']
            self.N_fc = result['N_fc']
        
        return np.array(P_out_up), np.array(P_out_down[::-1])
    
    def get_binary_state(self, threshold = 0.5):
        """Get binary state based on internal intensity."""
        # Normalize intensity
        I_norm = np.abs(self.a)**2 / (self.params.V_eff * 1e-12)  # Arbitrary normalization
        return 1 if I_norm > threshold else -1
