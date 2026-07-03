"""
Plot results directly in VS Code
No extra setup needed - VS Code shows plots automatically
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create results directory if it doesn't exist
Path("results").mkdir(exist_ok=True)
Path("results/plots").mkdir(exist_ok=True)

def plot_convergence():
    """Plot convergence of BO vs Random Search"""
    
    # Load results
    try:
        with open("results/real_bo_results.json", "r") as f:
            bo_results = json.load(f)
        bo_accuracy = [trial["accuracy"] for trial in bo_results["all_trials"]]
        bo_best = np.maximum.accumulate(bo_accuracy)
    except:
        print("BO results not found. Run python src/real_bo.py first")
        return
    
    try:
        with open("results/real_random_search.json", "r") as f:
            rs_results = json.load(f)
        rs_accuracy = [trial["accuracy"] for trial in rs_results["all_trials"]]
        rs_best = np.maximum.accumulate(rs_accuracy)
    except:
        print("Random Search results not found")
        rs_accuracy = []
        rs_best = []
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Individual trials
    axes[0, 0].plot(range(1, len(bo_accuracy)+1), bo_accuracy, 'bo-', alpha=0.7, label='BO', markersize=8)
    if rs_accuracy:
        axes[0, 0].plot(range(1, len(rs_accuracy)+1), rs_accuracy, 'rs-', alpha=0.7, label='Random Search', markersize=8)
    axes[0, 0].set_xlabel('Trial Number')
    axes[0, 0].set_ylabel('Validation Accuracy (%)')
    axes[0, 0].set_title('Individual Trial Performance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Best so far (convergence)
    axes[0, 1].plot(range(1, len(bo_best)+1), bo_best, 'b-', linewidth=2, label='BO', marker='o')
    if rs_best:
        axes[0, 1].plot(range(1, len(rs_best)+1), rs_best, 'r-', linewidth=2, label='Random Search', marker='s')
    axes[0, 1].set_xlabel('Number of Evaluations')
    axes[0, 1].set_ylabel('Best Accuracy (%)')
    axes[0, 1].set_title('Convergence: Best Accuracy Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate distribution
    bo_lrs = [trial["learning_rate"] for trial in bo_results["all_trials"]]
    axes[1, 0].hist(bo_lrs, bins=10, alpha=0.7, color='blue', label='BO')
    if rs_accuracy:
        rs_lrs = [trial["learning_rate"] for trial in rs_results["all_trials"]]
        axes[1, 0].hist(rs_lrs, bins=10, alpha=0.5, color='red', label='Random Search')
    axes[1, 0].set_xlabel('Learning Rate (log scale)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Learning Rate Distribution')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    
    # Plot 4: Accuracy vs Learning Rate
    axes[1, 1].scatter(bo_lrs, bo_accuracy, c='blue', label='BO', alpha=0.7, s=50)
    if rs_accuracy:
        axes[1, 1].scatter(rs_lrs, rs_accuracy, c='red', label='Random Search', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_title('Accuracy vs Learning Rate')
    axes[1, 1].set_xscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig("results/plots/convergence_comparison.png", dpi=150, bbox_inches='tight')
    print(" Plot saved to results/plots/convergence_comparison.png")
    
    # Show plot (will appear in VS Code)
    plt.show()

def plot_timeline():
    """Plot training time comparison"""
    
    try:
        with open("results/real_bo_results.json", "r") as f:
            bo_results = json.load(f)
        
        with open("results/real_random_search.json", "r") as f:
            rs_results = json.load(f)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract times (mock if not available)
        bo_times = [i+1 for i in range(len(bo_results["all_trials"]))]
        rs_times = [i+1 for i in range(len(rs_results["all_trials"]))]
        
        ax.plot(bo_times, [t["accuracy"] for t in bo_results["all_trials"]], 
                'b-', linewidth=2, label='BO')
        ax.plot(rs_times, [t["accuracy"] for t in rs_results["all_trials"]], 
                'r-', linewidth=2, label='Random Search')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Performance Comparison: BO vs Random Search')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/plots/performance_comparison.png", dpi=150)
        plt.show()
        
    except Exception as e:
        print(f"Could not create timeline plot: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Generating Plots for Research Paper")
    print("=" * 60)
    
    plot_convergence()
    plot_timeline()
    
    print("\n All plots saved to results/plots/")
    print(" View them in VS Code: Open results/plots/ folder")
