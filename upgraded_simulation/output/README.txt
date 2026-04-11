======================================================================
PEEK COMPOSITE SIMULATION RESULTS
Systematic Parametric Mapping of Multi-Filler PEEK Composites
======================================================================

Simulation Date: 2026-04-11 13:11:21
Report Generated: 2026-04-11 13:11:21

======================================================================
SIMULATION PARAMETERS
======================================================================
Matrix Material: PEEK
Matrix Modulus: 3.6 GPa
Matrix Thermal Conductivity: 0.25 W/mK
Volume Fractions: 0.0% to 30.0%
Number of Data Points: 20

======================================================================
FILLER MATERIALS
======================================================================

Carbon Fiber:
  Young's Modulus: 18.17 GPa (at 30%)
  Enhancement Factor: 5.05x
  Thermal Conductivity: 5.664 W/mK (at 30%)

Graphene Nanoplatelets:
  Young's Modulus: 137.07 GPa (at 30%)
  Enhancement Factor: 38.07x
  Thermal Conductivity: 280.747 W/mK (at 30%)

Glass Fiber:
  Young's Modulus: 8.96 GPa (at 30%)
  Enhancement Factor: 2.49x
  Thermal Conductivity: 0.475 W/mK (at 30%)

Carbon Nanotubes:
  Young's Modulus: 137.06 GPa (at 30%)
  Enhancement Factor: 38.07x
  Thermal Conductivity: 189.108 W/mK (at 30%)

======================================================================
HYBRID COMBINATIONS (at 30% total volume fraction)
======================================================================

Combination                                      E (GPa)             k (W/mK)
---------------------------------------------------------------------------
Graphene Nanoplatelets+Carbon Nanotubes           103.57              340.904
Carbon Fiber+Carbon Nanotubes                      75.30              163.051
Carbon Fiber+Graphene Nanoplatelets                75.10              168.089
Carbon Fiber+Graphene Nanoplatelets+Carbon Nanotubes           71.79              260.248
Graphene Nanoplatelets+Carbon Nanotubes+Glass Fiber           35.04              227.061
Graphene Nanoplatelets+Glass Fiber                 34.36               47.802
Carbon Nanotubes+Glass Fiber                       34.35               47.642
Carbon Fiber+Graphene Nanoplatelets+Carbon Nanotubes+Glass Fiber           27.68              183.223
Carbon Fiber+Graphene Nanoplatelets+Glass Fiber           25.87               66.678
Carbon Fiber+Carbon Nanotubes+Glass Fiber           25.76               66.250
Carbon Fiber+Glass Fiber                            8.54                0.882

======================================================================
MODELS IMPLEMENTED
======================================================================
Mechanical:
  - Rule of Mixtures (ROM)
  - Halpin-Tsai (with geometry-dependent xi)
  - Mori-Tanaka mean-field homogenization
  - Sequential Halpin-Tsai for hybrid systems

Thermal:
  - Maxwell-Garnett (spherical inclusions)
  - Nan's model (with interfacial thermal resistance)
  - Series/Parallel bounds
  - Sequential Nan model for hybrid systems

======================================================================
FILE STRUCTURE
======================================================================
/figures/png/          - High-resolution PNG files
/figures/pdf/          - Vector PDF files for publication
/figures/svg/          - Editable SVG files
/data/single_filler/   - CSV data for individual fillers
/data/hybrid_filler/   - CSV data for hybrid combinations
/data/summary/         - Summary tables and pivot tables

======================================================================
CITATION
======================================================================
If you use this data in your research, please cite:
[Author Names], 'Systematic parametric mapping of mechanical and
thermal properties in multi-filler PEEK composites', [Journal], [Year]

======================================================================
CONTACT
======================================================================
For questions or collaboration, contact:
[Author Email]
