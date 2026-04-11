# PEEK Multi-filler Composite Simulation Framework

This repository contains the complete Python implementation for mapping the mechanical and thermal property landscapes of multi-filler PEEK systems, as presented in our manuscript.

## Key Features
* **Micromechanical Models**: Implementation of Halpin-Tsai (mechanical) and Nan's Model (thermal) with interfacial resistance.
* **Hybrid Homogenization**: Sequential homogenization for quaternary systems (CF, GNP, CNT, GF).
* **Uncertainty Quantification**: Full Monte Carlo propagation (10,000 samples) for all input parameters.

## Reproducibility
To reproduce all figures and data presented in the paper, ensure you have the requirements installed and run the master script:

```bash
pip install -r requirements.txt
python main.py
