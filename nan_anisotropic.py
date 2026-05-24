"""
nan_anisotropic.py
Nan (1997) effective thermal conductivity model for randomly oriented
anisotropic spheroidal inclusions with directional Kapitza resistance.

Primary reference:
    Nan, C.-W., Birringer, R., Clarke, D.R., Gleiter, H. (1997).
    "Effective thermal conductivity of particulate composites with
    interfacial thermal resistance."
    J. Appl. Phys. 81(10), 6692–6699.
    DOI: 10.1063/1.365209

Supporting references:
    Chen, H. et al. (2016). "Thermal conductivity of polymer-based composites:
    Fundamentals and applications." Prog. Polym. Sci. 59, 41–85.
    DOI: 10.1016/j.progpolymsci.2016.03.001
    (Review that contextualises Nan's model for polymer composites)

    Kim, S.Y. et al. (2018). "Multiscale prediction of thermal conductivity for
    nanocomposites containing crumpled carbon nanofillers with interfacial characteristics."
    Compos. Sci. Technol. 155, 169–176.
    DOI: 10.1016/j.compscitech.2017.12.015
    (Multiscale Kapitza resistance calibration for PEEK nanocomposites)
"""

import numpy as np


def depolarization_factors(aspect_ratio: float,
                           geometry: str = 'fiber') -> tuple:
    """
    Depolarization factors (L_11, L_33) for a spheroidal inclusion.

    These factors control how the applied field is partitioned among
    the principal axes of the spheroid. They satisfy:
        2·L_11 + L_33 = 1          [Nan1997, Eq. 8]

    Parameters
    ----------
    aspect_ratio : float
        For fibers:    length / diameter  (p > 1, prolate)
        For platelets: diameter / thickness  (p > 1, oblate)
    geometry : str
        'fiber'    → prolate spheroid (symmetry axis = long dimension)
        'platelet' → oblate spheroid  (symmetry axis = short/thickness)

    Returns
    -------
    (L_11, L_33) : tuple of float
        L_33 is along the symmetry axis;
        L_11 is transverse (and L_11 = L_22 by symmetry).

    Limits (verification targets):
        Sphere  (p ≈ 1):  L_11 = L_33 = 1/3
        Fiber   (p → ∞):  L_33 → 0,  L_11 → 1/2
        Platelet(p → ∞):  L_33 → 1,  L_11 → 0

    Reference: [Nan1997] Eq. 8–10, DOI: 10.1063/1.365209
               Also presented in [Chen2016] Eq. 23–25,
               DOI: 10.1016/j.progpolymsci.2016.03.001
    """
    p = float(aspect_ratio)

    # --- Sphere limit ---
    if abs(p - 1.0) < 1e-6:
        return (1.0 / 3.0, 1.0 / 3.0)

    if geometry == 'fiber':
        # Prolate spheroid: semi-axes  a = a,  c = p·a  (c > a)
        # Eccentricity: e = sqrt(1 - 1/p²)
        # [Nan1997] Eq. 9 (prolate branch)
        e = np.sqrt(1.0 - 1.0 / p**2)
        L_33 = (1.0 - e**2) / (2.0 * e**3) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
        L_11 = (1.0 - L_33) / 2.0

    elif geometry == 'platelet':
        # Oblate spheroid: semi-axes  a = a,  c = a/p  (c < a)
        # The symmetry axis is the SHORT (thickness) direction.
        # [Nan1997] Eq. 10 (oblate branch)
        # With q = c/a = 1/p:
        #   e = sqrt(1 - q²)
        #   L_33 = (1/e²)(1 - q/e · arccos(q))     ... oblate form
        #   but equivalently using arctan:
        #   L_33 = p²/(p²-1) · [1 - arctan(sqrt(p²-1)) / sqrt(p²-1)]
        e_sq = 1.0 - 1.0 / p**2          # e² for oblate
        e = np.sqrt(e_sq)
        q = 1.0 / p                       # c/a ratio
        # Use the stable arctan form for oblate spheroid
        sqrt_term = np.sqrt(p**2 - 1.0)
        L_33 = (p**2 / (p**2 - 1.0)) * (1.0 - np.arctan(sqrt_term) / sqrt_term)
        L_11 = (1.0 - L_33) / 2.0

    else:
        raise ValueError(f"geometry must be 'fiber' or 'platelet', got '{geometry}'")

    return (L_11, L_33)


def nan_thermal_conductivity(V_f: float,
                             k_m: float,
                             k_f_axial: float,
                             k_f_transverse: float,
                             aspect_ratio: float,
                             geometry: str = 'fiber',
                             radius_nm: float = 5000.0,
                             a_K_nm: float = 0.1) -> float:
    """
    Nan (1997) Eq. 23: effective thermal conductivity of a composite
    with randomly oriented anisotropic spheroidal inclusions and
    interfacial (Kapitza) thermal resistance.

    Parameters
    ----------
    V_f : float
        Filler volume fraction (0–1).
    k_m : float
        Matrix thermal conductivity (W/mK).
        For PEEK: 0.25 W/mK  [Kurtz2019, DOI: 10.1016/C2016-0-02479-8]
                               [King2018, DOI: 10.1002/pc.24250]
    k_f_axial : float
        Filler conductivity along the symmetry axis (W/mK).
        Manuscript Table 1:
          CF:  50 W/mK   [King2018]
          GNP: 3000 W/mK (in-plane)  [King2018; Chen2016]
          CNT: 2000 W/mK (axial)     [Chen2016; Kim2018]
          GF:  1.0 W/mK  [Wang2020, DOI: 10.1016/j.compositesb.2020.108175]
    k_f_transverse : float
        Filler conductivity transverse to the symmetry axis (W/mK).
        Manuscript Table 1:
          CF:  5.0 W/mK   [King2018]
          GNP: 300 W/mK (through-thickness) [King2018; Chen2016]
          CNT: 20 W/mK (transverse)  [Chen2016; Kim2018]
          GF:  1.0 W/mK  [Wang2020]
    aspect_ratio : float
        Geometric aspect ratio (>1 for both fibers and platelets).
        CF: 20, GNP: 1000, CNT: 1000, GF: 15  [Manuscript Table 1]
    geometry : str
        'fiber' or 'platelet'. Determines depolarisation factor branch.
        CF → 'fiber', GNP → 'platelet', CNT → 'fiber', GF → 'fiber'
    radius_nm : float
        Inclusion cross-section radius (smallest characteristic half-dimension) in nanometres.
        Manuscript ¶46: "we set the inclusion radius r to 5 nm for CNTs,
        10 nm for GNPs (half-thickness), and 5 μm for CF and GF."
        CF: 5000 nm, GNP: 10 nm (half-thickness), CNT: 5 nm, GF: 5000 nm
    a_K_nm : float
        Kapitza radius = R_K · k_m (dimensionless, but quoted in nm).
        Manuscript ¶46, ¶61: "a typical Kapitza radius of a_K = 0.1 nm"
        [Kim2018, DOI: 10.1016/j.compscitech.2017.12.015]
        MC distribution: truncnorm, mean=0.1, CV=20%, bounds [0.05, 0.2]

    Returns
    -------
    k_eff : float
        Effective composite thermal conductivity (W/mK).
        Bounded below by k_m (physical floor).

    Theory
    ------
    [Nan1997] Eq. 23 for random orientation:

        k_eff = k_m · (3 + V_f · [2·β_11(1-L_11) + β_33(1-L_33)])
                     / (3 - V_f · [2·β_11·L_11    + β_33·L_33   ])

    where:
        β_ii = (k_c_ii - k_m) / (k_m + L_ii · (k_c_ii - k_m))

    and k_c_ii are the Kapitza-corrected directional conductivities:

        k_c_33 = k_f_axial     / (1 + k_f_axial     · a_K / (k_m · l_33))
        k_c_11 = k_f_transverse/ (1 + k_f_transverse · a_K / (k_m · l_11))

    Here l_33, l_11 are the characteristic lengths for heat flow along
    each principal axis (related to the filler dimensions).

    PLATELET RADIUS CONVENTION FIX:
    `radius_nm` is defined as the SMALLEST characteristic half-dimension.
    
    For fibers (prolate):
        l_11 = radius_nm                  (radius transverse to axis)
        l_33 = aspect_ratio · radius_nm   (half-length along symmetry axis)

    For platelets (oblate):
        l_33 = radius_nm                  (half-thickness along symmetry axis)
        l_11 = aspect_ratio · radius_nm   (in-plane radius)

    Reference: [Nan1997] Eq. 11–14, 23. DOI: 10.1063/1.365209
               [Chen2016] §3.2.           DOI: 10.1016/j.progpolymsci.2016.03.001
               [Kim2018] §2.              DOI: 10.1016/j.compscitech.2017.12.015
    """
    # --- Trivial cases ---
    if V_f <= 0.0:
        return k_m
    if V_f >= 1.0:
        # Purely filler — isotropic average (for safety)
        return (k_f_axial + 2.0 * k_f_transverse) / 3.0

    # --- Step 1: Depolarisation factors ---
    # [Nan1997] Eq. 8–10
    L_11, L_33 = depolarization_factors(aspect_ratio, geometry)

    # --- Step 2: Characteristic lengths for Kapitza correction ---
    # [Nan1997] Eq. 11–14; [Kim2018] §2
    # PLATELET RADIUS CONVENTION FIX applied here.
    if geometry == 'fiber':
        # Prolate: symmetry axis is the LONG axis
        l_33 = aspect_ratio * radius_nm   # half-length
        l_11 = radius_nm                  # radius
    elif geometry == 'platelet':
        # Oblate: symmetry axis is the SHORT (thickness) axis
        l_33 = radius_nm                  # half-thickness (smallest dimension)
        l_11 = aspect_ratio * radius_nm   # in-plane radius (largest dimension)
    else:
        raise ValueError(f"geometry must be 'fiber' or 'platelet', got '{geometry}'")

    # --- Step 3: Map conductivities to principal directions ---
    # For a FIBER:   symmetry axis (33) = LONG axis → k_f_axial goes to 33
    # For a PLATELET: symmetry axis (33) = SHORT axis (through-thickness)
    #                 → k_f_transverse (through-thickness) goes to 33
    #                 → k_f_axial (in-plane) goes to 11
    if geometry == 'platelet':
        k_f_33 = k_f_transverse   # through-thickness → symmetry axis
        k_f_11 = k_f_axial        # in-plane → transverse axis
    else:  # fiber
        k_f_33 = k_f_axial        # axial → symmetry axis
        k_f_11 = k_f_transverse   # transverse stays transverse

    # --- Step 4: Kapitza-corrected directional conductivities ---
    eps = 1e-30
    k_c_33 = k_f_33 / (1.0 + k_f_33 * a_K_nm / (k_m * max(l_33, eps)))
    k_c_11 = k_f_11 / (1.0 + k_f_11 * a_K_nm / (k_m * max(l_11, eps)))

    # --- Step 4: β factors ---
    # [Nan1997] Eq. 17: β_ii = (k_c_ii - k_m) / (k_m + L_ii·(k_c_ii - k_m))
    denom_33 = k_m + L_33 * (k_c_33 - k_m)
    denom_11 = k_m + L_11 * (k_c_11 - k_m)

    # Guard against zero denominators (degenerate cases)
    if abs(denom_33) < 1e-30:
        denom_33 = 1e-30
    if abs(denom_11) < 1e-30:
        denom_11 = 1e-30

    beta_33 = (k_c_33 - k_m) / denom_33
    beta_11 = (k_c_11 - k_m) / denom_11

    # --- Step 5: Nan Eq. 23 — random orientation average ---
    # k_eff/k_m = [3 + Vf·(2·β_11·(1-L_11) + β_33·(1-L_33))]
    #           / [3 - Vf·(2·β_11·L_11       + β_33·L_33     )]
    numerator = 3.0 + V_f * (2.0 * beta_11 * (1.0 - L_11) + beta_33 * (1.0 - L_33))
    denominator = 3.0 - V_f * (2.0 * beta_11 * L_11 + beta_33 * L_33)

    if abs(denominator) < 1e-30:
        # Fallback: parallel bound
        k_eff = (1.0 - V_f) * k_m + V_f * (k_f_axial + 2.0 * k_f_transverse) / 3.0
    else:
        k_eff = k_m * numerator / denominator

    # Physical floor: composite can't conduct less than neat matrix
    return max(k_m, k_eff)
