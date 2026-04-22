# hopfield-microring-resonators

# Photonic Hopfield Network with Microring Resonators

## Overview

This project explores a physical implementation of a Hopfield-like associative memory using nonlinear photonic devices, specifically microring resonators (MRRs) with carrier-induced dynamics.

The goal is to connect concepts from nonlinear optics, integrated photonics, and energy-based neural networks (Hopfield / Ising models), and demonstrate how optical systems can naturally perform optimization and memory retrieval.


## Physical Model

Each node is modeled as a nonlinear microring resonator with:

* optical field amplitude $a(t)$
* free carrier density $N(t)$

### Optical field dynamics

$$
\frac{da}{dt} = (i\Delta - \gamma)a + \sqrt{2\gamma_{ext}}, s_{in}
$$

### Carrier dynamics

$$
\frac{dN}{dt} = G(I) - \frac{N}{\tau_{fc}}
$$

* carriers are generated via two-photon absorption
* carrier density changes the refractive index, leading to a resonance shift


## Nonlinearity and Bistability

Carrier-induced refractive index change:

$$
\Delta n \propto N
$$

This leads to resonance shift and nonlinear feedback.

As a result, the system exhibits optical bistability:

* low intensity → state -1
* high intensity → state +1

Each resonator can therefore be interpreted as a binary neuron.


## Hopfield Energy Mapping

The classical Hopfield energy is defined as:

$$
E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j
$$

In the photonic system:

* optical coupling corresponds to interaction terms
* nonlinear response determines state updates
* time evolution of the system performs energy minimization


## Learning Rule

Patterns are stored using the Hebbian rule:

$$
W_{ij} = \frac{1}{N} \sum_p \xi_i^p \xi_j^p
$$


## Project Structure

```
hopfield_nn/
    network.py          # Hopfield network + optical simulation
    physics.py          # microring resonator model
    visualization.py    # plotting utilities

experiments/
    demo_single_recall.py
    exp_a_single_node.py
    exp_b_assoc_memory.py
    exp_c_physics.py
    exp_d_compare.py
    run_all.py
```


## How to Run

### Install dependencies

```bash
pip install numpy matplotlib scipy tqdm
```

### Run demo

```bash
python experiments/demo_single_recall.py
```

### Run full experiment suite

```bash
python experiments/run_all.py
```

---

## Features

* nonlinear microring resonator model
* carrier-induced bistability
* Hopfield associative memory
* pattern recall and noise robustness
* comparison with digital Hopfield networks


## Example Workflow

1. Generate binary patterns
2. Store them using Hebbian learning
3. Corrupt the input pattern
4. Run system dynamics (discrete or optical)
5. Observe convergence to a stored state


## Applications

* optical Ising machines
* neuromorphic photonics
* fast parallel optimization


