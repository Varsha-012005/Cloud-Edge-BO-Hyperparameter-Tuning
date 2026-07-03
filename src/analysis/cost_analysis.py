"""
AWS Cost Analysis for Research Paper
Compares different optimization strategies
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

class AWSCostAnalyzer:
    def __init__(self):
        self.costs = {}
        os.makedirs('experiments/costs', exist_ok=True)
        os.makedirs('report_graphs', exist_ok=True)
        os.makedirs('docs/tables', exist_ok=True)
        
        # AWS pricing (us-east-1, 2024)
        self.pricing = {
            't3.micro': 0.0104,
            't3.small': 0.0208,
            't3.medium': 0.0416,
            's3_storage': 0.023,
            's3_requests': 0.0004,
            'sqs_requests': 0.0000004,
            'data_transfer': 0.09
        }
        
    def calculate_sequential_cost(self, n_trials=12, time_per_trial=90):
        """Cost of sequential BO"""
        total_time_hours = (n_trials * time_per_trial) / 3600
        compute_cost = total_time_hours * self.pricing['t3.small']
        s3_cost = 0.01
        total = compute_cost + s3_cost
        
        return {
            'method': 'Sequential BO',
            'compute_hours': round(total_time_hours, 2),
            'compute_cost': round(compute_cost, 4),
            'total_cost': round(total, 4),
            'time_to_complete_hours': round(total_time_hours, 2)
        }
    
    def calculate_cloud_edge_cost(self, n_workers=3, n_batches=4, time_per_batch=90):
        """Cost of cloud-edge BO"""
        total_time_hours = (n_batches * time_per_batch) / 3600
        
        instance_costs = {
            'slow': self.pricing['t3.micro'] * total_time_hours,
            'medium': self.pricing['t3.small'] * total_time_hours,
            'fast': self.pricing['t3.medium'] * total_time_hours
        }
        
        compute_cost = sum(instance_costs.values())
        sqs_messages = n_workers * n_batches
        sqs_cost = sqs_messages * self.pricing['sqs_requests']
        s3_cost = 0.02
        data_transfer_gb = 0.1
        transfer_cost = data_transfer_gb * self.pricing['data_transfer']
        
        total = compute_cost + sqs_cost + s3_cost + transfer_cost
        
        return {
            'method': 'Cloud-Edge BO',
            'compute_hours': round(total_time_hours * n_workers, 2),
            'compute_cost': round(compute_cost, 4),
            'sqs_cost': round(sqs_cost, 4),
            'transfer_cost': round(transfer_cost, 4),
            'total_cost': round(total, 4),
            'time_to_complete_hours': round(total_time_hours, 2)
        }
    
    def calculate_batch_bo_cost(self, n_workers=3, n_batches=4, time_per_batch=90):
        """Cost of batch BO (non-cloud)"""
        total_time_hours = (n_batches * time_per_batch) / 3600
        compute_cost = total_time_hours * self.pricing['t3.medium']
        total = compute_cost
        
        return {
            'method': 'Batch BO',
            'compute_hours': round(total_time_hours, 2),
            'compute_cost': round(compute_cost, 4),
            'total_cost': round(total, 4),
            'time_to_complete_hours': round(total_time_hours, 2)
        }
    
    def analyze_all_methods(self):
        """Compare all methods"""
        print("\n" + "="*70)
        print("AWS COST ANALYSIS")
        print("For Different Optimization Strategies")
        print("="*70)
        
        methods = [
            self.calculate_sequential_cost(),
            self.calculate_batch_bo_cost(),
            self.calculate_cloud_edge_cost()
        ]
        
        with open('experiments/costs/cost_analysis.json', 'w') as f:
            json.dump(methods, f, indent=2)
        
        print("\nCost Comparison:")
        print("-" * 60)
        for method in methods:
            print(f"\n{method['method']}:")
            print(f"  Compute time: {method['compute_hours']:.2f} hours")
            print(f"  Compute cost: ${method['compute_cost']:.4f}")
            print(f"  Total cost: ${method['total_cost']:.4f}")
            print(f"  Time to complete: {method['time_to_complete_hours']:.2f} hours")
        
        seq_cost = methods[0]['total_cost']
        cloud_cost = methods[2]['total_cost']
        
        if seq_cost > 0 and cloud_cost > 0:
            if cloud_cost < seq_cost:
                savings = ((seq_cost - cloud_cost) / seq_cost) * 100
                print(f"\nCloud-Edge BO saves: {savings:.1f}% compared to Sequential BO")
            else:
                savings = ((cloud_cost - seq_cost) / cloud_cost) * 100
                speedup = methods[0]['time_to_complete_hours'] / methods[2]['time_to_complete_hours']
                print(f"\nCloud-Edge BO is {savings:.1f}% more expensive but {speedup:.1f}x faster")
        else:
            print("\nCloud-Edge BO is cost-effective")
        
        return methods
    
    def plot_cost_comparison(self):
        """Generate cost comparison plots"""
        methods = self.analyze_all_methods()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        names = [m['method'] for m in methods]
        costs = [m['total_cost'] for m in methods]
        
        bars = axes[0].bar(names, costs, color=['#E74C3C', '#F39C12', '#27AE60'])
        axes[0].set_ylabel('Total Cost (USD)', fontsize=12)
        axes[0].set_title('Cost Comparison of Optimization Methods', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        for bar, cost in zip(bars, costs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'${cost:.4f}', ha='center', fontsize=10)
        
        times = [m['time_to_complete_hours'] for m in methods]
        bars2 = axes[1].bar(names, times, color=['#E74C3C', '#F39C12', '#27AE60'])
        axes[1].set_ylabel('Time to Complete (hours)', fontsize=12)
        axes[1].set_title('Time Efficiency Comparison', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, t in zip(bars2, times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{t:.2f}h', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('report_graphs/cost_comparison.png', dpi=300, bbox_inches='tight')
        print("\nCost comparison plot saved to report_graphs/cost_comparison.png")
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        trials = range(1, 25)
        seq_costs = [self.calculate_sequential_cost(n_trials=t)['total_cost'] for t in trials]
        cloud_costs = []
        cloud_trials = []
        for t in trials:
            if t % 3 == 0:
                cloud_costs.append(self.calculate_cloud_edge_cost(n_batches=t//3)['total_cost'])
                cloud_trials.append(t)
        
        ax.plot(trials, seq_costs, 'r-', linewidth=2, label='Sequential BO')
        ax.plot(cloud_trials, cloud_costs, 'g-', linewidth=2, label='Cloud-Edge BO')
        ax.set_xlabel('Number of Trials', fontsize=12)
        ax.set_ylabel('Cumulative Cost (USD)', fontsize=12)
        ax.set_title('Cost Scaling with Number of Trials', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_graphs/cost_scaling.png', dpi=300, bbox_inches='tight')
        print("Cost scaling plot saved to report_graphs/cost_scaling.png")
        
        return methods
    
    def generate_cost_table_latex(self):
        """Generate LaTeX table for paper"""
        methods = self.analyze_all_methods()
        
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{AWS Cost Analysis for Different Optimization Methods}")
        latex.append("\\label{tab:costs}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\hline")
        latex.append("\\textbf{Method} & \\textbf{Compute Cost} & \\textbf{Total Cost} & \\textbf{Time (hours)} \\\\")
        latex.append("\\hline")
        
        for m in methods:
            latex.append(f"{m['method']} & ${m['compute_cost']:.4f} & ${m['total_cost']:.4f} & {m['time_to_complete_hours']:.2f} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        with open('docs/tables/cost_analysis.tex', 'w') as f:
            f.write("\n".join(latex))
        
        print("\nLaTeX cost table saved to docs/tables/cost_analysis.tex")
        return latex

if __name__ == "__main__":
    analyzer = AWSCostAnalyzer()
    analyzer.plot_cost_comparison()
    analyzer.generate_cost_table_latex()
    
    print("\n" + "="*70)
    print("COST ANALYSIS COMPLETE!")
    print("Results saved to experiments/costs/ and report_graphs/")
    print("="*70)
