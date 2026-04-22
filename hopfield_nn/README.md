# hopfield_nn — Core Photonic Hopfield Model

## Overview

This module implements the core components of a Hopfield-like associative memory using a physically-inspired photonic model based on microring resonators (MRRs).

It combines:

* Hopfield neural networks (energy-based models)
* nonlinear optical dynamics
* microring resonator physics

The implementation bridges abstract neural network models and realistic physical systems.

Recent research shows that photonic systems based on coupled resonators can emulate Hopfield-like Hamiltonian dynamics and collective behavior ([arXiv][1]), while microring resonators naturally provide nonlinearity and memory required for neural computation ([AIP Publishing][2]).


## Module Structure

```id="structure_block"
hopfield_nn/
    network.py          # Hopfield network logic + optical dynamics
    physics.py          # microring resonator model (MRR)
    visualization.py    # plotting and analysis tools
```


## Core Concepts

### 1. Hopfield Network

A Hopfield network is a recurrent system where each node is connected to every other node, and the system evolves to minimize an energy function:

$$
E = -\frac{1}{2} \sum_{i,j} W_{ij} s_i s_j
$$

The dynamics drive the system toward stable states (stored patterns), acting as associative memory ([Wikipedia][3]).


### 2. Photonic Mapping

In this implementation:

| Abstract model            | Physical system            |
| ------------------------- | -------------------------- |
| neuron $s_i \in {-1, +1}$ | microring resonator        |
| weight $W_{ij}$           | optical coupling           |
| state update              | nonlinear optical response |
| energy minimization       | time evolution             |

This mapping allows the network to be interpreted as a **physical dynamical system**, not just an algorithm.


### 3. Microring Resonator Model

Each node is modeled as a nonlinear resonator with:

* complex optical field $a(t)$
* carrier density $N(t)$

#### Optical dynamics

$$
\frac{da}{dt} = (i\Delta - \gamma)a + \sqrt{2\gamma_{ext}}, s_{in}
$$

#### Carrier dynamics

$$
\frac{dN}{dt} = G(I) - \frac{N}{\tau_{fc}}
$$

* nonlinear carrier effects shift resonance
* feedback produces bistability


### 4. Bistability as Binary State

Due to nonlinear feedback:

* low intensity → $-1$
* high intensity → $+1$

This allows each resonator to function as a **binary neuron**.

Microring resonators are particularly suitable for this because they combine:

* nonlinearity
* energy storage
* compact integration

which are key requirements for photonic neural systems ([arXiv][4]).


### 5. Learning Rule

Patterns are stored using Hebbian learning:

$$
W_{ij} = \frac{1}{N} \sum_p \xi_i^p \xi_j^p
$$


## Functionality

### `network.py`

* Hopfield weight matrix construction
* energy computation
* discrete update rules (async / sync)
* pattern recall
* optical simulation of network dynamics


### `physics.py`

* microring resonator model
* carrier-induced nonlinearity
* resonance shift and absorption
* time-domain simulation


### `visualization.py`

* state evolution plots
* energy landscape visualization
* pattern comparison


## Simulation Modes

### Discrete mode

* classical Hopfield dynamics
* baseline for comparison

### Optical mode

* continuous-time physical simulation
* nonlinear dynamics of coupled resonators
* more realistic but computationally heavier


## Design Philosophy

This module is not intended as a production neural network library.
Instead, it is a **research-oriented simulation framework** designed to:

* explore physical implementations of neural computation
* study nonlinear photonic dynamics
* investigate links between optics and optimization


## Relation to Current Research

Photonic neural networks and Ising machines are an active research area:

* microring-based networks can implement recurrent neural systems ([Nature][5])
* photonic systems can emulate energy-based optimization (Hopfield / Ising) ([Nature][6])

This module provides a simplified computational model of these ideas.


## Notes

* simplified physical model (no full EM simulation)
* intended for conceptual and algorithmic exploration
* suitable for extension toward experimental photonic systems


[1]: https://arxiv.org/abs/2603.04482?utm_source=chatgpt.com "Simulation of Hopfield-like Hamiltonians using time-multiplexed photonic networks"
[2]: https://pubs.aip.org/aip/app/article-pdf/doi/10.1063/5.0072090/19853795/051101_1_5.0072090.pdf?utm_source=chatgpt.com "Photonic and optoelectronic neuromorphic computing"
[3]: https://en.wikipedia.org/wiki/Hopfield_network?utm_source=chatgpt.com "Hopfield network"
[4]: https://arxiv.org/abs/2306.04779?utm_source=chatgpt.com "Photonic neural networks based on integrated silicon microresonators"
[5]: https://www.nature.com/articles/s41598-017-07754-z?utm_source=chatgpt.com "Neuromorphic photonic networks using silicon ..."
[6]: https://www.nature.com/articles/s41586-025-09838-7?utm_source=chatgpt.com "Programmable 200 GOPS Hopfield-inspired photonic Ising ..."

