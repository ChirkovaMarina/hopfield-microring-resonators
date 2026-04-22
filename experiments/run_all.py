"""Run the full photonic Hopfield experiment suite."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_a_single_node import ExpA_SingleNode
from experiments.exp_b_assoc_memory import ExpB_AssocRecall
from experiments.exp_c_physics import ExpC_Physics
from experiments.exp_d_compare import ExpD_Compare
from hopfield_nn.visualization import (
    visualize_energy_landscape,
    visualize_patterns,
    visualize_weight_matrix,
)


def run_all_experiments(output_dir = "results"):
    """Run all experiments and save results."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("Photonic Hopfield network tests")
    print("="*60)
    
    # -------------------------
    # Experiment A: Single Node
    # -------------------------
    print("\n" + "-"*40)
    print("EXPERIMENT A: Single MRR Characterization")
    print("-"*40)
    
    exp_a = ExpA_SingleNode()
    
    print("\nRunning hysteresis sweep...")
    exp_a.run_hysteresis()
    
    print("Running switching dynamics...")
    exp_a.run_switching()
    
    print("Running noise stability test...")
    exp_a.run_noise_stability(n_trials=30)
    
    fig_a = exp_a.plot_results(str(output_path / "experiment_A.png"))
    plt.close(fig_a)
    print(f"Results saved to {output_path / 'experiment_A.png'}")
    
    # -------------------------
    # Experiment B: Associative Recall
    # -------------------------
    print("\n" + "-"*40)
    print("EXPERIMENT B: Associative Memory")
    print("-"*40)
    
    exp_b = ExpB_AssocRecall(n_nodes=8)
    
    # Generate patterns
    patterns = exp_b.generate_random_patterns(2)
    print(f"\nStored patterns:\n{patterns}")
    
    print("\nRunning recall test...")
    exp_b.run_recall_test(patterns)
    
    print("Running capacity test...")
    exp_b.run_capacity_test()
    
    print("Running basin analysis...")
    exp_b.run_basin_analysis(patterns[0])
    
    fig_b = exp_b.plot_results(str(output_path / "experiment_B.png"))
    plt.close(fig_b)
    print(f"Results saved to {output_path / 'experiment_B.png'}")
    
    # Pattern visualization
    fig_pat = visualize_patterns(patterns)
    fig_pat.savefig(str(output_path / "stored_patterns.png"), dpi=150)
    plt.close(fig_pat)
    
    # Weight matrix visualization
    fig_w = visualize_weight_matrix(exp_b.network.W)
    fig_w.savefig(str(output_path / "weight_matrix.png"), dpi=150)
    plt.close(fig_w)
    
    # Energy landscape
    fig_e = visualize_energy_landscape(exp_b.network)
    if fig_e:
        fig_e.savefig(str(output_path / "energy_landscape.png"), dpi=150)
        plt.close(fig_e)
    
    # -------------------------
    # Experiment C: Physics Effects
    # -------------------------
    print("\n" + "-"*40)
    print("EXPERIMENT C: Physical Parameter Effects")
    print("-"*40)
    
    exp_c = ExpC_Physics(n_nodes=8)
    
    print("\nRunning Q factor sweep...")
    exp_c.run_q_factor_sweep()
    
    print("Running noise sweep...")
    exp_c.run_noise_sweep()
    
    print("Running asymmetry sweep...")
    exp_c.run_asymmetry_sweep()
    
    print("Creating stability map...")
    exp_c.create_stability_map(
        'amplitude_noise', np.linspace(0, 0.3, 10),
        'asymmetry', np.linspace(0, 0.4, 10),
        n_trials=15
    )
    
    fig_c = exp_c.plot_results(str(output_path / "experiment_C.png"))
    plt.close(fig_c)
    print(f"Results saved to {output_path / 'experiment_C.png'}")
    
    # -------------------------
    # Experiment D: Comparison
    # -------------------------
    print("\n" + "-"*40)
    print("EXPERIMENT D: Digital vs Photonic Comparison")
    print("-"*40)
    
    exp_d = ExpD_Compare(n_nodes=8)
    
    print("\nRunning recall comparison...")
    exp_d.compare_recall()
    
    print("Running noise robustness comparison...")
    exp_d.compare_noise_robustness()
    
    fig_d = exp_d.plot_results(str(output_path / "experiment_D.png"))
    plt.close(fig_d)
    print(f"Results saved to {output_path / 'experiment_D.png'}")
    
    # -------------------------
    # Summary
    # -------------------------
    print("\n" + "="*60)
    print("All tests finished")
    print("="*60)
    print(f"\nAll results saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob("*.png")):
        print(f"  - {f.name}")
    
    return {
        'exp_a': exp_a,
        'exp_b': exp_b,
        'exp_c': exp_c,
        'exp_d': exp_d
    }



if __name__ == "__main__":
    run_all_experiments("results")
