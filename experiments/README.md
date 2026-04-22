# experiments — Photonic Hopfield Experiments

## Overview

This directory contains a set of experiments designed to analyze the behavior of a photonic Hopfield-like network based on microring resonators (MRRs).

The experiments explore how nonlinear optical dynamics can reproduce key properties of associative memory and energy-based optimization.


## Structure

```id="exp_structure"
experiments/
    demo_single_recall.py     # simple example of pattern recall
    exp_a_single_node.py      # single resonator behavior
    exp_b_assoc_memory.py     # associative memory performance
    exp_c_physics.py          # effect of physical parameters
    exp_d_compare.py          # comparison with digital Hopfield
    run_all.py                # run all experiments
```


## Experiment A — Single Node Dynamics

Focus:

* optical bistability
* switching dynamics
* noise stability

This experiment studies a single microring resonator as a nonlinear system.

Key idea:

* carrier-induced nonlinearity leads to bistable behavior
* bistability enables binary state representation


## Experiment B — Associative Memory

Focus:

* pattern storage
* recall from corrupted inputs
* convergence behavior

This experiment demonstrates the core property of Hopfield networks:

* recovery of stored patterns from partial or noisy input


## Experiment C — Physical Effects

Focus:

* influence of physical parameters
* robustness to noise
* asymmetry and losses

Parameters studied:

* Q factor
* coupling strength
* noise levels

This experiment connects physical device properties to computational performance.


## Experiment D — Comparison with Digital Hopfield

Focus:

* recall accuracy
* convergence speed
* robustness to noise

This experiment compares:

* classical discrete Hopfield network
* photonic (continuous-time) implementation

Photonic systems can emulate neural dynamics through their natural time evolution, offering potential advantages in speed and parallelism ([Nature][3]).


## Demo

### Single pattern recall

```bash id="run_demo"
python experiments/demo_single_recall.py
```

Shows:

* corrupted input
* evolution of the network
* convergence to stored pattern


## Run Full Experiment Suite

```bash id="run_all"
python experiments/run_all.py
```

This will:

* run all experiments (A–D)
* generate plots
* save results

