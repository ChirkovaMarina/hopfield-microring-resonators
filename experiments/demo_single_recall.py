"""Small interactive demo for one associative recall run."""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hopfield_nn.network import PhotonicHopfieldNetwork
from hopfield_nn.visualization import visualize_recall_evolution


def demo_single_recall():
    """Demonstrate a single pattern recall with visualization."""
    print("Demo: single recall")
    print("-" * 40)
    
    # Create small network
    N = 8
    network = PhotonicHopfieldNetwork(N)
    
    # easy-to-see patterns
    pattern1 = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # First half +1
    pattern2 = np.array([1, -1, 1, -1, 1, -1, 1, -1])  # Alternating
    patterns = np.array([pattern1, pattern2])
    
    print(f"Pattern 1: {pattern1}")
    print(f"Pattern 2: {pattern2}")
    
    # Store patterns
    network.store_patterns(patterns)
    
    # corrupt one pattern a bit
    initial = pattern1.copy()
    initial[2] = -1  # Flip bit
    initial[5] = 1   # Flip bit
    print(f"\nCorrupted input: {initial}")
    print(f"(2 bits flipped from Pattern 1)")
    
    # Recall
    result = network.recall_discrete(initial)
    
    print(f"\nRecall result: {result['final_state']}")
    print(f"Converged in {result['iterations']} iterations")
    print(f"Matched pattern: {result['best_match'] + 1}")
    print(f"Success: {result['success']}")
    
    # Visualize
    fig = visualize_recall_evolution(result['history'], pattern1)
    plt.show()
    
    return network, result



if __name__ == "__main__":
    demo_single_recall()
