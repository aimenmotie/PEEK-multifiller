# Supplementary Information: Systematic Sensitivity Mapping

This document presents the detailed sensitivity analyses and numerical datasets supporting our manuscript on PEEK composite micromechanical modeling. All analysis scripts are included in the repository and can be executed to reproduce these tables and figures.

---

## Section S1: Homogenization Addition Order (Figure S1)
In multi-phase sequential homogenization schemes, the order in which reinforcement phases are mathematically compounded can theoretically affect the final composite properties. We compared three ordering schemes:
1. **Descending Aspect Ratio** (GNP/CNT $\rightarrow$ CF $\rightarrow$ GF)
2. **Ascending Aspect Ratio** (GF $\rightarrow$ CF $\rightarrow$ GNP/CNT)
3. **Random Order**

Our analysis of the quaternary system (CF + GNP + CNT + GF) showed that the sequential scheme is highly robust to ordering. The maximum detected variation in composite Young's modulus $E_c$ across all orders was **$4.8\%$** (occurring at the maximum $30\text{ vol\%}$ total loading in the CF+GNP+CNT system), confirming that the choice of sequencing does not introduce meaningful numerical bias into the predictions.

---

## Section S2: Equal Distribution Rule vs. Dirichlet Randomization (Figure S2)
To evaluate our assumption of equal volume fractions for multi-filler hybrids, we conducted a Monte Carlo sweep of $1,000$ random Dirichlet distribution vectors ($\sum V_{f,i} = V_{f,total}$) for the quaternary composite at $30\text{ vol\%}$ total loading. 
*   For the Dirichlet samples, the coefficient of variation (CV) was **$12\text{--}18\%$** for composite Young's modulus $E_c$, and **$15\text{--}22\%$** for composite thermal conductivity $k_c$.
*   The nominal equal-volume-fraction case lies almost exactly at the median of the randomized design landscape, confirming its utility as a representative metric for comparing multi-filler synergy.

---

## Section S3: Interfacial Bonding and Kapitza Sensitivity (Figure S3)
Interfacial thermal boundaries can be significantly altered by local defects and nonideal filler-matrix bonding. We expanded the Kapitza boundary radius $a_K$ over a broad sweep of $0.02\text{ nm}$ to $0.5\text{ nm}$ (based on [Kim2018], DOI: `10.1016/j.compscitech.2017.12.015`) to observe the impact on effective thermal conductivity $k_{eff}$. 
*   **Nano-reinforcements (GNP and CNT)** are highly sensitive to interfacial quality due to their extreme surface-area-to-volume ratio ($5\text{ nm}$ radius for CNT and $10\text{ nm}$ half-thickness for GNP). 
*   **Micro-reinforcements (CF and GF)** remain virtually unaffected by $a_K$ variations across the entire physical range due to their large characteristic dimensions ($5\,\mu\text{m}$ radius).

---

## Section S4: Carbon Nanotube (CNT) Sensitivity Sweeps (Figures S4–S6)
We conducted a comprehensive sensitivity analysis of the PEEK + CNT composite to isolate the impact of aspect ratio, interfacial thermal resistance, and loading/alignment parameters. 

### S4.1. Aspect Ratio Sensitivity (alpha: 10 to 2000)
*   **Young's Modulus**: Exhibits strong non-linear sensitivity to aspect ratio. The reinforcement efficiency begins to plateau around $\alpha \approx 1000$ (needle limit). Unidirectional (aligned) CNTs show a much steeper reinforcement rate compared to random 3D orientation.
*   **Thermal Conductivity**: Strongly benefits from increased aspect ratio under our proper Nan (1997) anisotropic model (Eq. 23, DOI: `10.1063/1.365209`). As aspect ratio increases, the axial heat flow component dominates ($L_{33} \rightarrow 0$), and the effective conductivity rises from $3.1\text{ W/mK}$ at $\alpha=10$ to $226.8\text{ W/mK}$ at $\alpha=2000$ ($15\text{ vol\%}$ loading).
*   **Data Source**: `output/data/summary/cnt_sensitivity_aspect_ratio.csv`
*   **Plot**: `output/figures/png/cnt_sensitivity_aspect_ratio.png`

### S4.2. Kapitza Interfacial Thermal Resistance Sensitivity ($a_K$: 0.0 to 1.0 nm)
*   At a fixed $15\text{ vol\%}$ loading, we swept the dimensionless Kapitza radius $a_K$ from $0.0\text{ nm}$ (perfect thermal contact) to $1.0\text{ nm}$ (highly insulating boundary).
*   The effective thermal conductivity of the PEEK-CNT composite drops precipitously from **$112.5\text{ W/mK}$** at $a_K = 0.0\text{ nm}$ to **$14.2\text{ W/mK}$** at $a_K = 1.0\text{ nm}$. This highlights that the thermal performance of nanocomposites is heavily interface-controlled, and validating $a_K$ is critical.
*   **Data Source**: `output/data/summary/cnt_sensitivity_kapitza.csv`
*   **Plot**: `output/figures/png/cnt_sensitivity_kapitza.png`

### S4.3. Volume Fraction & Orientation Sensitivity (V_f: 0% to 30%)
*   We swept CNT volume fraction from $0\text{ to }30\%$ comparing random 3D orientation and aligned unidirectional orientation.
*   **Modulus**: Under aligned orientation, $E_c$ reaches a maximum of **$153.2\text{ GPa}$** at $30\text{ vol\%}$, compared to **$78.4\text{ GPa}$** under random orientation.
*   **Thermal Conductivity**: Rises monotonically with volume fraction, showing a stable, non-linear curvature due to anisotropic Kapitza-corrected coupling.
*   **Data Source**: `output/data/summary/cnt_sensitivity_loading.csv`
*   **Plot**: `output/figures/png/cnt_sensitivity_loading.png`

---

## References
*   **[Nan1997]** Nan, C.-W., Birringer, R., Clarke, D.R., Gleiter, H. (1997). "Effective thermal conductivity of particulate composites with interfacial thermal resistance." *J. Appl. Phys.* 81(10), 6692–6699. DOI: `10.1063/1.365209`
*   **[TuckerLiang1999]** Tucker III, C.L. & Liang, E. (1999). "Stiffness predictions for unidirectional short-fiber composites: Review and evaluation." *Compos. Sci. Technol.* 59(5), 655–671. DOI: `10.1016/S0266-3538(98)00120-1`
*   **[Kim2018]** Kim, S.Y. et al. (2018). "Multiscale prediction of thermal conductivity for nanocomposites containing crumpled carbon nanofillers." *Compos. Sci. Technol.* 155, 169–176. DOI: `10.1016/j.compscitech.2017.12.015`
