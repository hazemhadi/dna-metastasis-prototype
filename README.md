
# DNA Metastasis Detection – BART Prototype (PyTorch)

This prototype simulates a research workflow for detecting metastatic cancer patterns in circulating DNA fragments using a modified BART model in PyTorch.

The goal is to show:

- How a **base model (BART)** can be **extended with a custom layer** for domain-specific pattern detection.
- How to **generate synthetic DNA fragment data** that mimics differences between healthy and metastatic profiles.
- How to build a **reproducible training + evaluation pipeline** with clear documentation and logging.

> **Note:** This project uses synthetic data only. It is a proof-of-concept demonstrating model design and scientific workflow – not a clinically validated tool.

## Project structure

```bash
dna-metastasis-bart-prototype/
│── data/                 # synthetic DNA fragment generation scripts
│── model/                # BART modification, custom layers, training code
│── notebooks/            # exploratory analysis & demo notebooks
│── results/              # metrics, logs, trained models
│── docs/                 # method notes, diagrams
│── requirements.txt
│── README.md
│── .gitignore
