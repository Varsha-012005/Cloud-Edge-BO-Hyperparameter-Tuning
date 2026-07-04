# Cloud-Edge Bayesian Optimization for Hyperparameter Tuning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Parallel Bayesian optimization framework for heterogeneous cloud-edge infrastructure.

## About this project

This project explores whether Bayesian Optimization can be made faster and smarter when the compute workers available aren't all the same speed. Most parallel BO methods assume every worker is equally capable, which almost never holds true in a real cloud-edge setup. Here, a cloud controller runs the Gaussian Process surrogate and hands out batches of configurations to three edge workers of different speeds, routing the more promising configurations to the faster ones so nothing sits idle.

## What's new here

- Parallel BO designed specifically for heterogeneous edge workers, not just equal-speed clusters
- A q-UCB batch acquisition strategy adapted for cloud-edge orchestration
- Hardware-aware evaluation that accounts for accuracy, runtime, CPU, and memory together, not accuracy alone

## Tools used

| Category | Tools |
|----------|-------|
| Cloud | AWS EC2 (hosts the GP surrogate model) |
| Training | Google Colab (CNN training) |
| IDE | VS Code |
| Containerization | Docker (used to simulate edge workers) |
| Version Control | GitHub |

## Datasets

| Dataset | Link | Reference |
|---------|------|-----------|
| MNIST | [Link](https://huggingface.co/datasets/ylecun/mnist) | LeCun et al. (1998) |
| Fashion-MNIST | [Link](https://github.com/zalandoresearch/fashion-mnist) | Xiao et al. (2017) |
| CIFAR-10 | [Link](https://www.cs.toronto.edu/~kriz/cifar.html) | Krizhevsky (2009) |

## Architecture

![Cloud-Edge Bayesian Optimization Architecture]<img width="600" height="650" alt="image" src="https://github.com/user-attachments/assets/395cd3ea-09a1-4fa7-8a32-9208a3d4852e" />


The cloud controller sits at the top and owns the Gaussian Process model, which predicts expected accuracy and how uncertain it is about that prediction. That feeds into the q-UCB acquisition step, which balances exploring new regions of the search space against exploiting what already looks promising. The batch proposer then generates three hyperparameter configurations at once, spread out enough to not waste evaluations on near-duplicates.

Each configuration goes into its own SQS queue, one per edge worker. The three workers aren't identical on purpose:

- Edge Worker 1 (Slow) — t3.micro, 1 vCPU, 1 GB RAM, roughly half the speed of the baseline
- Edge Worker 2 (Medium) — t3.small, 1 vCPU, 2 GB RAM, the baseline speed
- Edge Worker 3 (Fast) — t3.medium, 2 vCPU, 4 GB RAM, about twice the baseline speed

Every worker runs inside its own Docker container so results stay reproducible regardless of which machine picked up the job. Once training finishes, datasets, results, and logs all land in a shared S3 bucket that every worker and the controller can reach.

## Results so far

| Dataset | Baseline | Cloud-Edge BO | Improvement |
|---------|----------|---------------|-------------|
| MNIST | 98.50% | 99.58% | +1.08% |
| FashionMNIST | 89.96% | 92.79% | +2.83% |
| CIFAR-10 | 65.00% | 86.73% | +21.73% |

### Speedup from parallelizing

| Configuration | Time | Speedup |
|---------------|------|---------|
| Sequential (1 worker) | 36 minutes | 1x |
| Cloud-Edge (3 workers) | 12 minutes | 3x |

## Getting started

```bash
git clone https://github.com/Varsha-012005/Cloud-Edge-BO-Hyperparameter-Tuning.git
cd Cloud-Edge-BO-Hyperparameter-Tuning

python -m venv venv
.\venv\Scripts\activate

pip install -r requirements.txt
```

### Running experiments

```bash
# a quick sanity check with just a few trials
python src/core/sequential_bo.py --dataset mnist --trials 15

# the full run
python src/core/sequential_bo.py --dataset mnist --trials 15

# the actual cloud-edge method
python src/core/cloud_edge_bo.py
```

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
