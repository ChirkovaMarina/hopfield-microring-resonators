# hopfield-microring-resonators

Overview

This project explores a physical implementation of a Hopfield-like associative memory using nonlinear photonic devices — specifically microring resonators (MRRs) with carrier-induced dynamics.

The goal is to connect:

nonlinear optics
integrated photonics
energy-based neural networks (Hopfield / Ising models)

and demonstrate how physical optical systems can naturally perform optimization and memory retrieval.


Physical Model

Each node is modeled as a nonlinear microring resonator with:

optical field amplitude ( a(t) )
free carrier density ( N(t) )
Optical field dynamics

[
\frac{da}{dt} = (i\Delta - \gamma)a + \sqrt{2\gamma_{ext}}, s_{in}
]

Carrier dynamics

[
\frac{dN}{dt} = G(I) - \frac{N}{\tau_{fc}}
]

carriers are generated via two-photon absorption
carriers shift refractive index → resonance shift
Nonlinearity and Bistability

Carrier-induced refractive index change:

[
\Delta n \propto N
]

leads to resonance shift and nonlinear feedback.

This results in optical bistability:

low intensity → state -1
high intensity → state +1

Each resonator behaves like a binary neuron.

Hopfield Energy Mapping

The classical Hopfield energy:

[
E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j
]

is implicitly implemented through system dynamics:

optical coupling → interaction terms
nonlinear response → state update
time evolution → energy minimization


Learning Rule

Patterns are stored using Hebbian learning:

[
W_{ij} = \frac{1}{N} \sum_p \xi_i^p \xi_j^p
]

How to Run
Install dependencies
pip install numpy matplotlib scipy tqdm
Run demo
python experiments/demo_single_recall.py
Run all experiments
python experiments/run_all.py
Features
nonlinear microring resonator model
carrier-induced bistability
Hopfield associative memory
pattern recall and noise robustness
comparison with digital Hopfield networks
Example Workflow
Generate binary patterns
Store them using Hebbian learning
Corrupt input pattern
Run dynamics (discrete or optical)
Observe convergence to stored state
