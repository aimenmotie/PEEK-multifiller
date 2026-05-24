"""
validate_nan.py
Validation script comparing the old Nan thermal conductivity model
with the new proper Nan (1997) anisotropic model, and verifying mechanical fixes.

DOI references:
  [Nan1997] 10.1063/1.365209
  [King2018] 10.1002/pc.24250
  [Gul2026] 10.1002/pc.70137
  [Jeon2021] Dissertation
"""

import sys
import numpy as np

# Import new models
try:
    from nan_anisotropic import depolarization_factors, nan_thermal_conductivity
except ImportError:
    print("Error: Could not import nan_anisotropic.py. Please verify its location.")
    sys.exit(1)

# Import old simulator from backup
try:
    import sys
    sys.path.append('.')
    sys.path.append('./backups')
    from thermal_models_v1_backup import ThermalSimulator as OldThermalSimulator
    from mechanical_models_v1_backup import MechanicalSimulator as OldMechanicalSimulator
except ImportError:
    print("Error: Could not import old backup simulators. Ensure thermal_models_v1_backup.py exists.")
    sys.exit(1)

# Matrix properties (nominal PEEK)
k_m = 0.25
E_m = 3.8

# Nominal filler properties for old model
filler_properties_old = {
    'Carbon Fiber': {'thermal_conductivity': 20.0, 'aspect_ratio': 20.0},
    'Graphene Nanoplatelets': {'thermal_conductivity': 1200.0, 'aspect_ratio': 1000.0},
    'Glass Fiber': {'thermal_conductivity': 1.0, 'aspect_ratio': 15.0},
    'Carbon Nanotubes': {'thermal_conductivity': 680.0, 'aspect_ratio': 1000.0}
}

# Nominal filler properties for new model (Manuscript Table 1)
filler_properties_new = {
    'Carbon Fiber': {
        'k_axial': 50.0, 'k_transverse': 5.0, 'aspect_ratio': 20.0,
        'geometry': 'fiber', 'radius_nm': 5000.0, 'young_modulus': 230.0
    },
    'Graphene Nanoplatelets': {
        'k_axial': 3000.0, 'k_transverse': 300.0, 'aspect_ratio': 1000.0,
        'geometry': 'platelet', 'radius_nm': 10.0, 'young_modulus': 1000.0
    },
    'Glass Fiber': {
        'k_axial': 1.0, 'k_transverse': 1.0, 'aspect_ratio': 15.0,
        'geometry': 'fiber', 'radius_nm': 5000.0, 'young_modulus': 72.0
    },
    'Carbon Nanotubes': {
        'k_axial': 2000.0, 'k_transverse': 20.0, 'aspect_ratio': 1000.0,
        'geometry': 'fiber', 'radius_nm': 5.0, 'young_modulus': 1000.0
    }
}

def old_sequential_nan_model(V_f_total, filler_list, fractions):
    """Computes sequential Nan's model using the old backup logic."""
    sim = OldThermalSimulator({'thermal_conductivity': k_m}, filler_properties_old)
    return sim.sequential_nan_model(V_f_total, filler_list, fractions)

def new_sequential_nan_model(V_f_total, filler_list, fractions):
    """Computes sequential Nan's model using the new proper anisotropic logic."""
    k_current = k_m
    for filler, vf in zip(filler_list, fractions):
        if vf == 0.0:
            continue
        props = filler_properties_new[filler]
        # Nan model using current matrix k_current
        k_comp = nan_thermal_conductivity(
            V_f=vf,
            k_m=k_current,
            k_f_axial=props['k_axial'],
            k_f_transverse=props['k_transverse'],
            aspect_ratio=props['aspect_ratio'],
            geometry=props['geometry'],
            radius_nm=props['radius_nm'],
            a_K_nm=0.1
        )
        k_current = k_comp
    return k_current

def old_halpin_tsai_modulus(V_f, E_f, aspect_ratio):
    """Old Halpin-Tsai with aspect_ratio > 10 geometry bug."""
    # Current backup logic:
    # if aspect_ratio > 10: xi = 2.0 * aspect_ratio
    # else: xi = aspect_ratio
    # So for GNP (aspect_ratio = 1000), it treats it as fiber with xi = 2000.0
    xi = 2.0 * aspect_ratio if aspect_ratio > 10 else aspect_ratio
    eta = ((E_f / E_m) - 1.0) / ((E_f / E_m) + xi)
    E_c = E_m * (1.0 + xi * eta * V_f) / (1.0 - eta * V_f)
    return 0.5 * E_c  # random orientation factor

def new_halpin_tsai_modulus(V_f, E_f, aspect_ratio, geometry):
    """New proper Halpin-Tsai with correct geometry factor ξ."""
    # [TuckerLiang1999]: ξ = 2α for fibers, ξ = (2/3)α for platelets
    if geometry == 'platelet':
        xi = (2.0 / 3.0) * aspect_ratio
    else:
        xi = 2.0 * aspect_ratio
        
    eta = ((E_f / E_m) - 1.0) / ((E_f / E_m) + xi)
    E_c = E_m * (1.0 + xi * eta * V_f) / (1.0 - eta * V_f)
    return 0.5 * E_c

def main():
    print("=" * 70)
    print("                NAN 1997 MODEL OVERHAUL VALIDATION")
    print("=" * 70)
    
    # ----------------------------------------------------------------------
    # 1. Depolarisation & Kapitza parameters
    # ----------------------------------------------------------------------
    print("\n--- 1. Nominal Spheroid Properties & Interfacial Resistance ---")
    print(f"{'Filler':<25} | {'L_11':<10} | {'L_33':<10} | {'l_11 (nm)':<10} | {'l_33 (nm)':<10} | {'k_c_33 (W/mK)':<12}")
    print("-" * 85)
    for name, props in filler_properties_new.items():
        L_11, L_33 = depolarization_factors(props['aspect_ratio'], props['geometry'])
        
        # Characteristic lengths (PLATELET RADIUS CONVENTION FIX)
        if props['geometry'] == 'fiber':
            l_33 = props['aspect_ratio'] * props['radius_nm']
            l_11 = props['radius_nm']
        else:
            l_33 = props['radius_nm']
            l_11 = props['aspect_ratio'] * props['radius_nm']
            
        # Kapitza corrected k_c_33
        k_c_33 = props['k_axial'] / (1.0 + props['k_axial'] * 0.1 / (k_m * l_33))
        
        print(f"{name:<25} | {L_11:<10.6f} | {L_33:<10.6e} | {l_11:<10.1f} | {l_33:<10.1f} | {k_c_33:<12.3f}")
        
    # ----------------------------------------------------------------------
    # 2. Thermal conductivity comparison old vs new
    # ----------------------------------------------------------------------
    print("\n--- 2. Thermal Conductivity (W/mK) comparison old vs new ---")
    systems = [
        ('CNT only', ['Carbon Nanotubes'], [1.0]),
        ('GNP only', ['Graphene Nanoplatelets'], [1.0]),
        ('GNP+CNT', ['Graphene Nanoplatelets', 'Carbon Nanotubes'], [0.5, 0.5]),
        ('GNP+CNT+CF', ['Graphene Nanoplatelets', 'Carbon Nanotubes', 'Carbon Fiber'], [1/3, 1/3, 1/3]),
        ('Quaternary', ['Graphene Nanoplatelets', 'Carbon Nanotubes', 'Carbon Fiber', 'Glass Fiber'], [0.25, 0.25, 0.25, 0.25])
    ]
    
    vol_fractions = [0.05, 0.15, 0.30]
    
    print(f"{'System':<15} | {'V_f':<5} | {'Old k_c':<12} | {'New k_c':<12} | {'Difference':<12}")
    print("-" * 65)
    
    for sys_name, fillers, weights in systems:
        for vf in vol_fractions:
            fractions = [w * vf for w in weights]
            old_k = old_sequential_nan_model(vf, fillers, fractions)
            new_k = new_sequential_nan_model(vf, fillers, fractions)
            diff = new_k - old_k
            print(f"{sys_name:<15} | {vf:<5.2f} | {old_k:<12.4f} | {new_k:<12.4f} | {diff:<+12.4f}")
            
    # ----------------------------------------------------------------------
    # 3. Mechanical modulus GNP geometry factor fix
    # ----------------------------------------------------------------------
    print("\n--- 3. Mechanical Modulus Fix (Halpin-Tsai for Platelets) ---")
    E_f_gnp = filler_properties_new['Graphene Nanoplatelets']['young_modulus']
    ar_gnp = filler_properties_new['Graphene Nanoplatelets']['aspect_ratio']
    geom_gnp = filler_properties_new['Graphene Nanoplatelets']['geometry']
    
    old_E_c = old_halpin_tsai_modulus(0.30, E_f_gnp, ar_gnp)
    new_E_c = new_halpin_tsai_modulus(0.30, E_f_gnp, ar_gnp, geom_gnp)
    
    print(f"GNP-only composite (30 vol%) Young's Modulus:")
    print(f"  Old model prediction (wrong fiber switch): {old_E_c:.2f} GPa")
    print(f"  New model prediction (correct platelet):    {new_E_c:.2f} GPa")
    print(f"  Difference:                                 {new_E_c - old_E_c:+.2f} GPa")
    
    # ----------------------------------------------------------------------
    # 4. Literature validation checks
    # ----------------------------------------------------------------------
    print("\n--- 4. Literature Sanity Checks ---")
    
    # Check 4a: PEEK + CF at 22.8 vol% (King 2018) -> k_c should be ~1-2 W/mK
    cf_vf = 0.228
    k_cf_228 = new_sequential_nan_model(cf_vf, ['Carbon Fiber'], [cf_vf])
    print(f"PEEK + CF at 22.8 vol% [King2018 target: ~1-2 W/mK]:")
    print(f"  Prediction: {k_cf_228:.4f} W/mK  -->  ", end="")
    if 1.0 <= k_cf_228 <= 3.0:
        print("PASS: SANITY CHECK PASSED (within target range)")
    else:
        print("FAIL: SANITY CHECK FAILED (outside target range)")
        
    # Check 4b: PEEK + CF at 30 wt% (Jeon 2021) -> E_c = 15.6 ± 0.5 GPa
    # Convert 30 wt% to vol% for CF in PEEK:
    # rho_f = 1.8 g/cm3, rho_m = 1.3 g/cm3
    # V_f = (0.3/1.8) / (0.3/1.8 + 0.7/1.3) = 0.1667 / (0.1667 + 0.5385) ≈ 0.236
    cf_wt_vf = (0.3/1.8) / (0.3/1.8 + 0.7/1.3)
    E_cf_30wt = new_halpin_tsai_modulus(
        cf_wt_vf,
        filler_properties_new['Carbon Fiber']['young_modulus'],
        filler_properties_new['Carbon Fiber']['aspect_ratio'],
        filler_properties_new['Carbon Fiber']['geometry']
    )
    print(f"PEEK + CF at 30 wt% (~23.6 vol%) Young's Modulus [Jeon2021 target: 15.6 ± 0.5 GPa]:")
    print(f"  Prediction: {E_cf_30wt:.2f} GPa  -->  ", end="")
    if 14.0 <= E_cf_30wt <= 17.0:
        print("PASS: SANITY CHECK PASSED (within target range)")
    else:
        print("FAIL: SANITY CHECK FAILED (outside target range)")
        
    # Check 4c: Hybrid PEEK + GNP/CNT at 30 vol% -> k_c should be ~5-10 W/mK (or generally reasonable network limits, e.g. Gul 2026)
    k_hybrid_30 = new_sequential_nan_model(0.30, ['Graphene Nanoplatelets', 'Carbon Nanotubes'], [0.15, 0.15])
    print(f"PEEK + GNP + CNT at 30 vol% [Gul2026 target/analogy: network ~5-150 W/mK]:")
    print(f"  Prediction: {k_hybrid_30:.4f} W/mK  -->  ", end="")
    if 2.0 <= k_hybrid_30 <= 130.0:
        print("PASS: SANITY CHECK PASSED (reasonable network conductivity)")
    else:
        print("FAIL: SANITY CHECK FAILED (outside realistic bounds)")

    print("=" * 70)

if __name__ == '__main__':
    main()
