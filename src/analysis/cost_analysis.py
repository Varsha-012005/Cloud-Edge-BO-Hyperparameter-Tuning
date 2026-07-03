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
        
        # AWS pricing (us-east-1, 2024)
        self.pricing = {
            't3.micro': 0.0104,   # per hour
            't3.small': 0.0208,   # per hour
            't3.medium': 0.0416,  # per hour
            's3_storage': 0.023,   # per GB-month
            's3_requests': 0.0004,  # per 1000 requests
            'sqs_requests': 0.0000004,  # per request
            'data_transfer': 0.09  # per GB
        }
        
    def calculate_sequential_cost(self, n_trials=12, time_per_trial=90):
        """Cost of sequential BO"""
        total_time_hours = (n_trials * time_per_trial) / 3600
        
        # Single instance (t3.small)
        compute_cost = total_time_hours * self.pricing['t3.small']
        
        # S3 storage (minimal)
        s3_cost = 0.01
        
        total = compute_cost + s3_cost
        
        return {
            'method': 'Sequential BO',
            'compute_hours': total_time_hours,
            'compute_cost': compute_cost,
            'total_cost': total,
            'time_to_complete_hours': total_time_hours
        }
    
    def calculate_cloud_edge_cost(self, n_workers=3, n_batches=4, time_per_batch=90):
        """Cost of cloud-edge BO"""
        total_time_hours = (n_batches * time_per_batch) / 3600
        
        # Multiple instances (heterogeneous)
        instance_costs = {
            'slow': self.pricing['t3.micro'] * total_time_hours,
            'medium': self.pricing['t3.small'] * total_time_hours,
            'fast': self.pricing['t3.medium'] * total_time_hours
        }
        
        compute_cost = sum(instance_costs.values())
        
        # SQS costs
        sqs_messages = n_workers * n_batches
        sqs_cost = sqs_messages * self.pricing['sqs_requests']
        
        # S3 costs
        s3_cost = 0.05  # Storage + results
        
        # Data transfer (results files)
        data_transfer_gb = 0.5
        transfer_cost = data_transfer_gb * self.pricing['data_transfer']
        
        total = compute_cost + sqs_cost + s3_cost + transfer_cost
        
        return {
            'method': 'Cloud-Edge BO',
            'compute_hours': total_time_hours * n_workers,
            'compute_cost': compute_cost,
            'sqs_cost': sqs_cost,
            'transfer_cost': transfer_cost,
            'total_cost': total,
            'time_to_complete_hours': total_time_hours
        }
    
    def calculate_batch_bo_cost(self, n_workers=3, n_batches=4, time_per_batch=90):
        """Cost of batch BO (non-cloud)"""
        total_time_hours = (n_batches * time_per_batch) / 3600
        
        # Single fast instance (t3.medium)
        compute_cost = total_time_hours * self.pricing['t3.medium']
        
        total = compute_cost
        
        return {
            'method': 'Batch BO',
            'compute_hours': total_time_hours,
            'compute_cost': compute_cost,
            'total_cost': total,
            'time_to_complete_hours': total_time_hours
        }
    
    def analyze_all_methods(self):
        """Compare all methods"""
        print("\n" + "="*70)
        print("💰 AWS COST ANALYSIS")
        print("For Different Optimization Strategies")
        print("="*70)
        
        methods = [
            self.calculate_sequential_cost(),
            self.calculate_batch_bo_cost(),
            self.calculate_cloud_edge_cost()
        ]
        
        # Save results
        with open('experiments/costs/cost_analysis.json', 'w') as f:
            json.dump(methods, f, indent=2)
        
        # Print comparison
        print("\n📊 Cost Comparison:")
        print("-" * 60)
        for method in methods:
            print(f"\n{method['method']}:")
            print(f"  Compute time: {method['compute_hours']:.2f} hours")
            print(f"  Compute cost: ${method['compute_cost']:.2f}")
            print(f"  Total cost: ${method['total_cost']:.2f}")
            print(f"  Time to complete: {method['time_to_complete_hours']:.2f} hours")
        
        # Calculate savings
        seq_cost = methods[0]['total_cost']
        cloud_cost = methods[2]['total_cost']
        savings = ((seq_cost - cloud_cost) / seq_cost) * 100
        
        print(f"\n💡 Cloud-Edge BO saves: {savings:.1f}% compared to Sequential BO")
        
        return methods, savings
    
    def plot_cost_comparison(self):
        """Generate cost comparison plots"""
        methods, savings = self.analyze_all_methods()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Total cost
        names = [m['method'] for m in methods]
        costs = [m['total_cost'] for m in methods]
        
        bars = axes[0].bar(names, costs, color=['red', 'orange', 'green'])
        axes[0].set_ylabel('Total Cost (USD)', fontsize=12)
        axes[0].set_title('Cost Comparison of Optimization Methods', fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'${cost:.2f}', ha='center', fontsize=10)
        
        # Plot 2: Time to complete
        times = [m['time_to_complete_hours'] for m in methods]
        
        bars2 = axes[1].bar(names, times, color=['red', 'orange', 'green'])
        axes[1].set_ylabel('Time to Complete (hours)', fontsize=12)
        axes[1].set_title('Time Efficiency Comparison', fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for bar, t in zip(bars2, times):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{t:.1f}h', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('report_graphs/cost_comparison.png', dpi=300)
        print("\n✅ Cost comparison plot saved to report_graphs/cost_comparison.png")
        
        # Plot 3: Savings over trials
        fig, ax = plt.subplots(figsize=(10, 6))
        
        trials = range(1, 25)
        seq_costs = [self.calculate_sequential_cost(n_trials=t)['total_cost'] for t in trials]
        cloud_costs = [self.calculate_cloud_edge_cost(n_batches=t//3)['total_cost'] for t in trials if t%3==0]
        
        ax.plot(trials, seq_costs, 'r-', linewidth=2, label='Sequential BO')
        ax.plot(range(3, 25, 3), cloud_costs, 'g-', linewidth=2, label='Cloud-Edge BO')
        ax.set_xlabel('Number of Trials', fontsize=12)
        ax.set_ylabel('Cumulative Cost (USD)', fontsize=12)
        ax.set_title('Cost Scaling with Number of Trials', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_graphs/cost_scaling.png', dpi=300)
        print("✅ Cost scaling plot saved to report_graphs/cost_scaling.png")
        
        return methods
    
    def generate_cost_table_latex(self):
        """Generate LaTeX table for paper"""
        methods, savings = self.analyze_all_methods()
        
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
            latex.append(f"{m['method']} & ${m['compute_cost']:.2f} & ${m['total_cost']:.2f} & {m['time_to_complete_hours']:.1f} \\\\")
        
        latex.append("\\hline")
        latex.append(f"\\multicolumn{{4}}{{l}}{{\\footnotesize Savings: Cloud-Edge BO saves {savings:.1f}\\% vs Sequential}} \\\\")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        with open('docs/tables/cost_analysis.tex', 'w') as f:
            f.write("\n".join(latex))
        
        print("\n✅ LaTeX cost table saved to docs/tables/cost_analysis.tex")
        return latex

if __name__ == "__main__":
    analyzer = AWSCostAnalyzer()
    analyzer.plot_cost_comparison()
    analyzer.generate_cost_table_latex()
    
    print("\n" + "="*70)
    print("✅ COST ANALYSIS COMPLETE!")
    print("📁 Results saved to experiments/costs/ and report_graphs/")
    print("="*70)