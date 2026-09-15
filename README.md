# PINNsFormer-Comprehensive-PDE-Benchmark


**A study and evaluation of PINNsFormer, a Transformer-based Physics-Informed Neural Network, for solving Partial Differential Equations — compared against the standard PINN baseline.**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

## About

This repository contains the implementation and results of my bachelor's project at **Amirkabir University of Technology (Tehran Polytechnic)**, Faculty of Mathematics and Computer Science, supervised by **Dr. Fatemeh Shakeri**.

The project reproduces **PINNsFormer** ([Zhao, Ding & Prakash, ICLR 2024](https://arxiv.org/abs/2312.10529)) — a Transformer-based architecture for Physics-Informed Neural Networks that converts each spatio-temporal input point into a short pseudo-sequence and processes it with self-attention — and evaluates it against a standard MLP-based PINN baseline on five benchmark PDEs:

| Equation | Order | Domain |
|---|---|---|
| Burgers' Equation | 2nd | $x \in [-1,1],\ t \in [0,1]$ |
| Allen–Cahn Equation | 2nd | $x \in [-1,1],\ t \in [0,1]$ |
| Nonlinear Schrödinger Equation | 2nd (complex) | $x \in [-5,5],\ t \in [0,\pi/2]$ |
| Kuramoto–Sivashinsky Equation | 4th | $x \in [-10,10],\ t \in [0,1]$ |
| Korteweg–de Vries (KdV) Equation | 3rd | $x \in [-1,1],\ t \in [0,1]$ |

> **Note:** The Allen–Cahn notebook is part of the completed project (see the report) but is not yet uploaded to this repository — it will be added soon.

## Key Idea

Standard PINNs feed each point $(x, t)$ to the network independently, with no explicit notion of temporal dependency between nearby time steps. **PINNsFormer** addresses this by:

1. **Pseudo-sequence generation** — expanding each point $(x, t)$ into a short sequence $[(x, t_0), (x, t_1), \dots, (x, t_{k-1})]$ over $k$ nearby time steps.
2. **Spatio-temporal mixing** — projecting each 2D point into a higher-dimensional embedding.
3. **Attention-based Encoder–Decoder** — learning dependencies between sequence elements via self-attention (à la [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).
4. **Wavelet activation** — a learnable $\omega_1 \sin(x) + \omega_2 \cos(x)$ activation, used in place of Tanh, with better expressivity for oscillatory solutions.

Both models are trained with L-BFGS (strong Wolfe line search) on a composite physics-informed loss (residual + boundary + initial condition terms), using automatic differentiation for all PDE derivatives.

## Results

Relative error (rMAE) of PINNsFormer vs. the standard PINN baseline, evaluated against reference numerical/analytical solutions:

| Equation | PINN (rMAE) | PINNsFormer (rMAE) | Improvement |
|---|---:|---:|---:|
| Burgers' | 0.0262 | 0.0120 | ~54% |
| Allen–Cahn | 0.9910 | 0.3790 | ~62% |
| Nonlinear Schrödinger | 0.0601 | 0.0032 | ~95% |
| Kuramoto–Sivashinsky | 0.3012 | 0.0001 | >99% |
| KdV | 0.0820 | 0.0184 | ~78% |

PINNsFormer outperformed the standard PINN on every equation tested, with the largest gain on the 4th-order Kuramoto–Sivashinsky equation. This improvement in accuracy came at a computational cost: PINNsFormer's training time was, on average, roughly **13× longer** than the standard PINN's across all experiments.

Full derivations, architecture diagrams, per-equation error curves, prediction heatmaps, and a detailed discussion of failure modes and limitations are available in the [project report](report/PINNsFormer_Bachelor_Report_FA.pdf) (in Persian).

## Repository Structure

```
PINNsFormer-PDE-Evaluation/
├── notebooks/
│   ├── PINNsFormer_Burgers.ipynb        # Burgers' equation
│   ├── PINNsFormer_KdV.ipynb            # Korteweg-de Vries equation
│   ├── PINNsFormer_KS.ipynb             # Kuramoto-Sivashinsky equation
│   ├── PINNsFormer_Schrodinger.ipynb    # Nonlinear Schrodinger equation
│   └── PINNsFormer_AllenCahn.ipynb      # Allen-Cahn equation (coming soon)
├── report/
│   └── PINNsFormer_Bachelor_Report_FA.pdf   # Full bachelor's project report (Persian)
├── assets/                              # Figures used in this README
├── requirements.txt
├── LICENSE
└── README.md
```

Each notebook is self-contained and follows the same pipeline:

1. Define collocation, boundary, and initial-condition points for the PDE domain.
2. Build the pseudo-sequence representation for PINNsFormer's attention mechanism.
3. Train a standard PINN baseline (4-layer MLP, Tanh activation) with L-BFGS.
4. Train PINNsFormer (encoder–decoder with self-attention, Wavelet activation) with the same optimizer and iteration budget.
5. Evaluate both models against a reference solution (rMAE / rRMSE) and visualize predictions vs. absolute error.

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

### Running a notebook

```bash
git clone https://github.com/MariaMahdavi/PINNsFormer-PDE-Evaluation.git
cd PINNsFormer-PDE-Evaluation
jupyter notebook notebooks/PINNsFormer_Burgers.ipynb
```

Each notebook automatically downloads its reference solution (`.mat` file) from the [original PINN repository](https://github.com/maziarraissi/PINNs) and runs both models end-to-end — no extra setup required. A GPU is recommended but not required (training will fall back to CPU automatically).

## References

This project is a reproduction and evaluation study built on top of the following works:

- Zhao, Z., Ding, X., & Prakash, B. A. (2024). *PINNsFormer: A Transformer-Based Framework for Physics-Informed Neural Networks.* ICLR.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686–707.
- Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS.
- Rahaman, N., et al. (2019). *On the spectral bias of neural networks.* ICML.
- Krishnapriyan, A., et al. (2021). *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS.

The full reference list is included in the [project report](report/PINNsFormer_Bachelor_Report_FA.pdf).

## Authors

- **Maria Mahdavi Lain** — [GitHub](https://github.com/MariaMahdavi)
- **Zohreh Sarmali Saeedi**

Bachelor's Project, Amirkabir University of Technology (Tehran Polytechnic), Faculty of Mathematics and Computer Science
Supervisor: Dr. Fatemeh Shakeri — Mordad 1405 (August 2026)

## License

This project is released under the [MIT License](LICENSE).
