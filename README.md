# Cloud-Edge Bayesian Optimization for Hyperparameter Tuning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Parallel Bayesian optimization framework for heterogeneous cloud-edge infrastructure.

## About this project

This project explores whether Bayesian Optimization can be made faster and smarter when the compute workers available aren't all the same speed. Most parallel BO methods assume every worker is equally capable, which almost never holds true in a real cloud-edge setup. Here, a cloud controller runs the Gaussian Process surrogate and hands out batches of configurations to three edge workers of different speeds, routing the more promising configurations to the faster ones so nothing sits idle.

The framework achieves **3× speedup** over sequential execution by parallelizing hyperparameter evaluations across three heterogeneous EC2 workers (t3.micro, t3.small, t3.medium).

## What's new here

- Parallel BO designed specifically for heterogeneous edge workers, not just equal-speed clusters
- A q-UCB batch acquisition strategy adapted for cloud-edge orchestration
- Worker-speed-aware acquisition function with heterogeneity compensation
- Hardware-aware evaluation that accounts for accuracy, runtime, CPU, and memory together
- Fully reproducible Docker-based deployment across AWS EC2 instances

## Tools used

| Category | Tools |
|----------|-------|
| Cloud | AWS EC2, S3, SQS |
| Training | PyTorch, TorchVision |
| IDE | VS Code |
| Containerization | Docker |
| Version Control | GitHub |
| GP Surrogate | GPyTorch, Scikit-learn |

## Datasets

| Dataset | Link | Reference |
|---------|------|-----------|
| MNIST | [Link](https://huggingface.co/datasets/ylecun/mnist) | LeCun et al. (1998) |
| Fashion-MNIST | [Link](https://github.com/zalandoresearch/fashion-mnist) | Xiao et al. (2017) |
| CIFAR-10 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) | Krizhevsky (2009) |

## Architecture

![Cloud-Edge Bayesian Optimization Architecture]<img width="1024" height="1536" alt="figure1 (2)" src="https://github.com/user-attachments/assets/cf76dd2c-043b-43b1-9cbe-41c5fea0392c" />


The cloud controller sits at the top and owns the Gaussian Process model, which predicts expected accuracy and how uncertain it is about that prediction. That feeds into the q-UCB acquisition step, which balances exploring new regions of the search space against exploiting what already looks promising. The batch proposer then generates three hyperparameter configurations at once, spread out enough to not waste evaluations on near-duplicates.

Each configuration goes into its own SQS queue, one per edge worker. The three workers aren't identical on purpose:

| Worker | Instance | vCPU | RAM | Speed Factor |
|--------|----------|------|-----|--------------|
| Edge Worker 1 (Slow) | t3.micro | 1 | 1 GB | 0.5× |
| Edge Worker 2 (Medium) | t3.small | 1 | 2 GB | 1.0× |
| Edge Worker 3 (Fast) | t3.medium | 2 | 4 GB | 2.0× |

Every worker runs inside its own Docker container so results stay reproducible regardless of which machine picked up the job. Once training finishes, datasets, results, and logs all land in a shared S3 bucket that every worker and the controller can reach.

## Results

### Classification Accuracy

| Dataset | Baseline | Cloud-Edge BO (Best Seed) | Cloud-Edge BO (5-Seed Mean ± Std) |
|---------|----------|---------------------------|-----------------------------------|
| MNIST | 98.50% | **99.31%** | 99.19% ± 0.13% |
| FashionMNIST | 89.96% | **89.09%** | 88.08% ± 0.86% |
| CIFAR-10 | 65.00% | **74.46%** | 70.83% ± 3.36% |

### Speedup from Parallelization

| Configuration | Time | Speedup | Efficiency |
|---------------|------|---------|------------|
| Sequential (1 worker) | 36 minutes | 1× | 100% |
| Cloud-Edge (3 workers) | **12 minutes** | **3×** | **93%** |

### Best Hyperparameters Found

| Dataset | Learning Rate | Batch Size | Epochs | Accuracy |
|---------|---------------|------------|--------|----------|
| MNIST | 0.002389 | 64 | 4 | 99.31% |
| FashionMNIST | 0.002406 | 64 | 4 | 89.09% |
| CIFAR-10 | 0.000745 | 16 | 4 | 74.46% |

## Getting started

```bash
git clone https://github.com/Varsha-012005/Cloud-Edge-BO-Hyperparameter-Tuning.git
cd Cloud-Edge-BO-Hyperparameter-Tuning

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt

### Generating graphs

```bash
python src/analysis/generate_graphs.py
```

## Researcher

| Name | Institution | GitHub |
|------|-------------|--------|
| Varsha | D.Y. Patil International University | [@Varsha-012005](https://github.com/Varsha-012005) |
| Dr. Swati Vijay Shinde | Pimpri Chinchwad College of Engineering | Supervisor |

## License

MIT License

## Contact

GitHub: [@Varsha-012005](https://github.com/Varsha-012005)
