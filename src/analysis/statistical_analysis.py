import numpy as np
import json
import os
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class StatisticalAnalyzer:
    def __init__(self):
        self.results = {}
        # Create all required directories
        os.makedirs('report_graphs', exist_ok=True)
        os.makedirs('docs/figures', exist_ok=True)
        os.makedirs('docs/tables', exist_ok=True)
        
        # Set font to avoid emoji issues
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        
    def load_results_from_folder(self):
        """Automatically load all result files from results folder"""
        result_files = {
            'Sequential BO MNIST': 'results/sequential_bo_mnist.json',
            'Sequential BO Fashion': 'results/sequential_bo_fashionmnist.json',
            'Sequential BO CIFAR10': 'results/sequential_bo_cifar10.json',
        }
        
        for name, filepath in result_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        if 'all_trials' in data:
                            accuracies = [trial['accuracy'] for trial in data['all_trials']]
                        elif 'all_results' in data:
                            accuracies = [r['accuracy'] for r in data['all_results']]
                        else:
                            accuracies = [data.get('best_accuracy', 0)]
                        
                        self.results[name] = accuracies
                        print(f"[OK] Loaded {name}: {len(accuracies)} trials, best={max(accuracies):.2f}%")
                except Exception as e:
                    print(f"[WARN] Could not load {name}: {e}")
        
        return self.results
    
    def friedman_test(self):
        """Friedman test for multiple methods"""
        print("\n" + "="*70)
        print("FRIEDMAN TEST - Statistical Significance")
        print("="*70)
        
        if len(self.results) < 2:
            print("[ERROR] Need at least 2 methods for comparison")
            return None, None
        
        max_len = max(len(v) for v in self.results.values())
        aligned_results = []
        
        for accuracies in self.results.values():
            if len(accuracies) < max_len:
                accuracies = accuracies + [accuracies[-1]] * (max_len - len(accuracies))
            aligned_results.append(accuracies)
        
        statistic, p_value = friedmanchisquare(*aligned_results)
        
        print(f"\nResults:")
        print(f"   Chi-squared: {statistic:.4f}")
        print(f"   P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"   [SIGNIFICANT] difference (p < 0.05)")
        else:
            print(f"   [NOT SIGNIFICANT] (p >= 0.05)")
        
        return statistic, p_value
    
    def wilcoxon_pairwise(self):
        """Pairwise comparisons"""
        print("\n" + "="*70)
        print("PAIRWISE COMPARISONS (Wilcoxon Test)")
        print("="*70)
        
        methods = list(self.results.keys())
        results_list = []
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                m1, m2 = methods[i], methods[j]
                
                min_len = min(len(self.results[m1]), len(self.results[m2]))
                d1 = self.results[m1][:min_len]
                d2 = self.results[m2][:min_len]
                
                _, p_value = wilcoxon(d1, d2)
                
                mean1, mean2 = np.mean(d1), np.mean(d2)
                better = m1 if mean1 > mean2 else m2
                improvement = abs(mean1 - mean2)
                
                results_list.append({
                    'Method A': m1,
                    'Method B': m2,
                    'Mean A (%)': f"{mean1:.2f}",
                    'Mean B (%)': f"{mean2:.2f}",
                    'Improvement': f"{improvement:.2f}%",
                    'p-value': f"{p_value:.6f}",
                    'Significant': "YES" if p_value < 0.05 else "NO",
                    'Winner': better
                })
                
                print(f"\n{m1} vs {m2}:")
                print(f"   Means: {mean1:.2f}% vs {mean2:.2f}%")
                print(f"   Improvement: {improvement:.2f}%")
                print(f"   p-value: {p_value:.6f}")
                print(f"   Significant: {'YES' if p_value < 0.05 else 'NO'}")
                print(f"   Winner: {better}")
        
        return pd.DataFrame(results_list)
    
    def calculate_effect_sizes(self):
        """Calculate Cohen's d effect sizes"""
        print("\n" + "="*70)
        print("EFFECT SIZES (Cohen's d)")
        print("="*70)
        
        methods = list(self.results.keys())
        
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                m1, m2 = methods[i], methods[j]
                
                min_len = min(len(self.results[m1]), len(self.results[m2]))
                d1 = self.results[m1][:min_len]
                d2 = self.results[m2][:min_len]
                
                mean_diff = np.mean(d1) - np.mean(d2)
                pooled_std = np.sqrt((np.std(d1)**2 + np.std(d2)**2) / 2)
                cohens_d = abs(mean_diff / pooled_std) if pooled_std > 0 else 0
                
                if cohens_d < 0.2:
                    interpretation = "Negligible"
                elif cohens_d < 0.5:
                    interpretation = "Small"
                elif cohens_d < 0.8:
                    interpretation = "Medium"
                else:
                    interpretation = "Large"
                
                print(f"\n{m1} vs {m2}:")
                print(f"   Cohen's d = {cohens_d:.3f} ({interpretation} effect)")
        
        return True
    
    def convergence_speed_analysis(self):
        """Analyze how fast each method converges"""
        print("\n" + "="*70)
        print("CONVERGENCE SPEED ANALYSIS")
        print("="*70)
        
        for name, accuracies in self.results.items():
            best = max(accuracies)
            target_95 = best * 0.95
            
            trials_95 = len(accuracies)
            for i, acc in enumerate(accuracies):
                if acc >= target_95 and trials_95 == len(accuracies):
                    trials_95 = i + 1
            
            print(f"\n{name}:")
            print(f"   Best: {best:.2f}%")
            print(f"   Reaches 95% of best in {trials_95} trials")
        
        return True
    
    def generate_paper_plots(self):
        """Generate publication-quality plots - FIXED for Matplotlib 3.9+"""
        print("\n" + "="*70)
        print("GENERATING PAPER-READY FIGURES")
        print("="*70)
        
        # Figure 1: Convergence curves
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for idx, (name, accuracies) in enumerate(self.results.items()):
            x = range(1, len(accuracies) + 1)
            best_so_far = np.maximum.accumulate(accuracies)
            ax.plot(x, best_so_far, 'o-', linewidth=2, markersize=6, 
                   label=name, color=colors[idx % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Number of Evaluations', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Convergence of Bayesian Optimization on Different Datasets', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('report_graphs/convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('docs/figures/convergence_comparison.pdf', bbox_inches='tight')
        print("[OK] Figure 1: Convergence comparison saved")
        
        # Figure 2: Box plot comparison - FIXED for Matplotlib 3.9
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = [self.results[name] for name in self.results.keys()]
        # FIXED: Changed 'labels' to 'tick_labels' for Matplotlib 3.9+
        bp = ax.boxplot(data_to_plot, tick_labels=list(self.results.keys()), patch_artist=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors[:len(self.results)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Accuracies Across Methods', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('report_graphs/boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig('docs/figures/boxplot_comparison.pdf', bbox_inches='tight')
        print("[OK] Figure 2: Box plot comparison saved")
        
        # Figure 3: Improvement heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        methods = list(self.results.keys())
        n_methods = len(methods)
        improvement_matrix = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(n_methods):
                if i != j:
                    mean_i = np.mean(self.results[methods[i]])
                    mean_j = np.mean(self.results[methods[j]])
                    improvement_matrix[i, j] = mean_i - mean_j
        
        im = ax.imshow(improvement_matrix, cmap='RdBu_r', vmin=-10, vmax=10)
        ax.set_xticks(range(n_methods))
        ax.set_yticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_title('Improvement Matrix (% better than column)', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(n_methods):
            for j in range(n_methods):
                text = ax.text(j, i, f'{improvement_matrix[i, j]:+.1f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, label='Improvement (%)')
        plt.tight_layout()
        plt.savefig('report_graphs/improvement_heatmap.png', dpi=300, bbox_inches='tight')
        print("[OK] Figure 3: Improvement heatmap saved")
        
        plt.close('all')
        return True
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper submission"""
        print("\n" + "="*70)
        print("GENERATING LATEX TABLES")
        print("="*70)
        
        # Table 1: Summary statistics
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Performance Comparison of Optimization Methods}")
        latex.append("\\label{tab:performance}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\hline")
        latex.append("\\textbf{Method} & \\textbf{Best (\\%)} & \\textbf{Mean (\\%)} & \\textbf{Std} & \\textbf{Trials} \\\\")
        latex.append("\\hline")
        
        for name, accuracies in self.results.items():
            best = max(accuracies)
            mean = np.mean(accuracies)
            std = np.std(accuracies)
            n = len(accuracies)
            latex.append(f"{name} & {best:.2f} & {mean:.2f} & {std:.3f} & {n} \\\\")
        
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        with open('docs/tables/performance_table.tex', 'w') as f:
            f.write("\n".join(latex))
        print("[OK] Table 1: Performance table saved to docs/tables/performance_table.tex")
        
        # Table 2: Statistical test results
        latex2 = []
        latex2.append("\\begin{table}[t]")
        latex2.append("\\centering")
        latex2.append("\\caption{Statistical Significance Testing Results}")
        latex2.append("\\label{tab:statistics}")
        latex2.append("\\begin{tabular}{llcccc}")
        latex2.append("\\hline")
        latex2.append("\\textbf{Method A} & \\textbf{Method B} & \\textbf{Mean A} & \\textbf{Mean B} & \\textbf{p-value} & \\textbf{Signif.} \\\\")
        latex2.append("\\hline")
        
        methods = list(self.results.keys())
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                m1, m2 = methods[i], methods[j]
                min_len = min(len(self.results[m1]), len(self.results[m2]))
                d1 = self.results[m1][:min_len]
                d2 = self.results[m2][:min_len]
                _, p_value = wilcoxon(d1, d2)
                mean1, mean2 = np.mean(d1), np.mean(d2)
                sig = "$p < 0.05$" if p_value < 0.05 else "n.s."
                
                latex2.append(f"{m1} & {m2} & {mean1:.2f}\\% & {mean2:.2f}\\% & {p_value:.4f} & {sig} \\\\")
        
        latex2.append("\\hline")
        latex2.append("\\end{tabular}")
        latex2.append("\\end{table}")
        
        with open('docs/tables/statistical_table.tex', 'w') as f:
            f.write("\n".join(latex2))
        print("[OK] Table 2: Statistical table saved to docs/tables/statistical_table.tex")
        
        return True
    
    def generate_report(self):
        """Generate complete analysis report"""
        print("\n" + "="*70)
        print("GENERATING COMPLETE ANALYSIS REPORT")
        print("="*70)
        
        report = []
        report.append("# Statistical Analysis Report for Cloud-Edge Bayesian Optimization")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## 1. Results Summary")
        
        for name, accuracies in self.results.items():
            report.append(f"\n### {name}")
            report.append(f"- Number of trials: {len(accuracies)}")
            report.append(f"- Best accuracy: {max(accuracies):.2f}%")
            report.append(f"- Mean accuracy: {np.mean(accuracies):.2f}%")
            report.append(f"- Standard deviation: {np.std(accuracies):.3f}")
        
        # Add Friedman test
        stat, p = self.friedman_test()
        if stat:
            report.append("\n## 2. Friedman Test (Global Comparison)")
            report.append(f"- Chi-squared statistic: {stat:.4f}")
            report.append(f"- P-value: {p:.6f}")
            if p < 0.05:
                report.append("- **Conclusion**: Significant differences exist between datasets")
            else:
                report.append("- **Conclusion**: No significant differences detected")
        
        # Save report
        with open('docs/statistical_report.md', 'w') as f:
            f.write("\n".join(report))
        
        print("\n[OK] Complete report saved to docs/statistical_report.md")
        
        # Print summary
        print("\n" + "="*70)
        print("FINAL SUMMARY FOR PAPER")
        print("="*70)
        for name, accuracies in self.results.items():
            print(f"\n{name}:")
            print(f"  Best: {max(accuracies):.2f}%")
            print(f"  Mean ± Std: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.3f}")
        
        return report

# Run the analysis
if __name__ == "__main__":
    print("="*70)
    print("STATISTICAL ANALYSIS FOR RESEARCH PAPER")
    print("Cloud-Edge Bayesian Optimization")
    print("="*70)
    
    analyzer = StatisticalAnalyzer()
    
    # Load your results
    analyzer.load_results_from_folder()
    
    if analyzer.results:
        # Run all analyses
        analyzer.friedman_test()
        analyzer.wilcoxon_pairwise()
        analyzer.calculate_effect_sizes()
        analyzer.convergence_speed_analysis()
        
        # Generate outputs
        analyzer.generate_paper_plots()
        analyzer.generate_latex_tables()
        analyzer.generate_report()
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("Check these folders:")
        print("   - report_graphs/     (PNG figures)")
        print("   - docs/figures/      (PDF figures)")
        print("   - docs/tables/       (LaTeX tables)")
        print("   - docs/statistical_report.md (Full report)")
        print("="*70)
    else:
        print("\n[ERROR] No result files found!")