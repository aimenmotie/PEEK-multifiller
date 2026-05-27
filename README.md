# PEEK Multi-filler Composite Simulation Framework

This repository contains the complete Python implementation for mapping the mechanical and thermal property landscapes of multi-filler PEEK systems, as presented in our manuscript.

## Key Features
* **Micromechanical Models**: Implementation of Halpin-Tsai (mechanical) and Nan (1997) anisotropic model (thermal) with Kapitza interfacial resistance.
* **Hybrid Homogenization**: Sequential homogenization for quaternary systems (CF, GNP, CNT, GF).
* **Uncertainty Quantification**: Full Monte Carlo propagation (10,000 samples) for all input parameters.
* **Validation Suite**: Literature comparison with both idealized and commercial-grade filler properties.
* **Sensitivity Analysis**: CNT aspect ratio, Kapitza radius, loading/alignment sweeps.

## Reproducibility
To reproduce all figures and data presented in the paper, ensure you have the requirements installed and run the master script:

```bash
pip install -r requirements.txt
python main.py
```

## Validation
Run the validation suite to verify model correctness and literature agreement:

```bash
python test_nan_unit.py      # 11 unit tests (physics limits, boundary conditions)
python validate_nan.py       # Literature validation + internal consistency
python cnt_sensitivity.py    # CNT sensitivity sweeps (Figures S4-S6)
python generate_supplementary.py  # Supplementary figures S1-S3
```

## File Structure
| File | Description |
|------|-------------|
| `main.py` | Master orchestration script |
| `mechanical_models.py` | Halpin-Tsai and sequential homogenization |
| `thermal_models.py` | Nan model, Maxwell, Bruggeman with Monte Carlo |
| `nan_anisotropic.py` | Nan (1997) Eq. 23 for anisotropic spheroids |
| `microstructure.py` | RVE microstructure visualization |
| `plotting.py` | Publication-quality figure generation |
| `data_export.py` | CSV and figure export |
| `test_nan_unit.py` | Unit test suite (11 tests) |
| `validate_nan.py` | Literature validation and consistency checks |
| `cnt_sensitivity.py` | CNT sensitivity analysis |
| `generate_supplementary.py` | Supplementary figures S1-S3 |
