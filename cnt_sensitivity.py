"""
cnt_sensitivity.py
Carbon Nanotube (CNT) sensitivity analysis for PEEK-CNT composites.
Sweeps aspect ratio, Kapitza interfacial resistance, and loading/alignment
to assess prediction sensitivity of Young's Modulus and Thermal Conductivity.
Saves figures to output/figures/png/ and data to output/data/summary/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from thermal_models import ThermalSimulator
from mechanical_models import MechanicalSimulator

# Ensure output directories exist
os.makedirs('output/figures/png', exist_ok=True)
os.makedirs('output/figures/pdf', exist_ok=True)
os.makedirs('output/data/summary', exist_ok=True)

# Define properties consistent with main.py
matrix_properties = {
    'young_modulus': 3.6,           # GPa
    'tensile_strength': 100,        # MPa
    'thermal_conductivity': 0.25,   # W/mK [Kurtz2019]
    'poisson_ratio': 0.40
}

filler_properties = {
    'Carbon Nanotubes': {
        'young_modulus': 1000.0,
        'tensile_strength': 10000.0,
        'thermal_conductivity': 680.0,
        'k_axial': 2000.0,
        'k_transverse': 20.0,
        'aspect_ratio': 1000.0,
        'geometry': 'fiber',
        'radius_nm': 5.0,
        'poisson_ratio': 0.20
    }
}

# Initialize simulators
mech_sim = MechanicalSimulator(matrix_properties, filler_properties)
therm_sim = ThermalSimulator(matrix_properties, filler_properties)

def apply_academic_style(ax):
    """Applies clean academic styling to plot axes."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, labelsize=10)
    ax.grid(True, linestyle='--', alpha=0.5, color='#cbd5e1')

def run_sensitivity_analysis():
    print("=" * 60)
    print("           CNT SENSITIVITY ANALYSIS GENERATION")
    print("=" * 60)

    # ----------------------------------------------------------------------
    # Sweep 1: Aspect Ratio Sensitivity
    # ----------------------------------------------------------------------
    print("1. Running aspect ratio sensitivity sweep (alpha: 10 to 10000)...")
    ar_sweep = np.linspace(10, 10000, 100)
    V_f_fixed = 0.15  # 15 vol% loading
    
    E_random = np.zeros_like(ar_sweep)
    E_aligned = np.zeros_like(ar_sweep)
    k_eff = np.zeros_like(ar_sweep)
    
    for idx, ar in enumerate(ar_sweep):
        E_random[idx] = mech_sim.halpin_tsai(
            V_f=V_f_fixed, E_f=1000.0, aspect_ratio=ar, orientation='random', geometry='fiber'
        )
        E_aligned[idx] = mech_sim.halpin_tsai(
            V_f=V_f_fixed, E_f=1000.0, aspect_ratio=ar, orientation='aligned', geometry='fiber'
        )
        # Thermal: using k_axial=2000, k_transverse=20, radius_nm=5, a_k=0.1 nm
        k_eff[idx] = therm_sim.nan_model(
            V_f=V_f_fixed, k_f=2000.0, aspect_ratio=ar, a_k=0.1,
            k_f_transverse=20.0, geometry='fiber', radius_nm=5.0
        )
        
    df_ar = pd.DataFrame({
        'Aspect_Ratio': ar_sweep,
        'Youngs_Modulus_Random_GPa': E_random,
        'Youngs_Modulus_Aligned_GPa': E_aligned,
        'Thermal_Conductivity_WperMk': k_eff
    })
    df_ar.to_csv('output/data/summary/cnt_sensitivity_aspect_ratio.csv', index=False)
    print("   Saved data to: output/data/summary/cnt_sensitivity_aspect_ratio.csv")
    
    # Plot 1: Aspect Ratio sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.plot(ar_sweep, E_random, color='#c0392b', linewidth=2.5, label='Random Orientation')
    ax1.plot(ar_sweep, E_aligned, color='#34495e', linewidth=2, linestyle='--', label='Aligned Orientation')
    ax1.set_xlabel('CNT Aspect Ratio (L/d)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Composite Young's Modulus (GPa)", fontsize=11, fontweight='semibold')
    ax1.set_title("Modulus Sensitivity (V_f = 15%)", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(frameon=False)
    apply_academic_style(ax1)
    
    ax2.plot(ar_sweep, k_eff, color='#2980b9', linewidth=2.5)
    ax2.set_xlabel('CNT Aspect Ratio (L/d)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Effective Thermal Conductivity (W/mK)', fontsize=11, fontweight='semibold')
    ax2.set_title("Thermal Sensitivity (V_f = 15%)", fontsize=12, fontweight='bold', pad=10)
    apply_academic_style(ax2)
    
    plt.tight_layout()
    fig.savefig('output/figures/png/cnt_sensitivity_aspect_ratio.png', dpi=300, bbox_inches='tight')
    fig.savefig('output/figures/pdf/cnt_sensitivity_aspect_ratio.pdf', bbox_inches='tight')
    print("   Saved plot to output/figures/png/cnt_sensitivity_aspect_ratio.png")
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Sweep 2: Interfacial Thermal Resistance (Kapitza Radius a_K) Sensitivity
    # ----------------------------------------------------------------------
    print("2. Running interfacial resistance sensitivity sweep (a_K: 0.0 to 1.0 nm)...")
    a_K_sweep = np.linspace(0.0, 1.0, 100)
    k_eff_ak = np.zeros_like(a_K_sweep)
    
    for idx, ak in enumerate(a_K_sweep):
        k_eff_ak[idx] = therm_sim.nan_model(
            V_f=V_f_fixed, k_f=2000.0, aspect_ratio=1000.0, a_k=ak,
            k_f_transverse=20.0, geometry='fiber', radius_nm=5.0
        )
        
    df_ak = pd.DataFrame({
        'Kapitza_Radius_nm': a_K_sweep,
        'Thermal_Conductivity_WperMk': k_eff_ak
    })
    df_ak.to_csv('output/data/summary/cnt_sensitivity_kapitza.csv', index=False)
    print("   Saved data to: output/data/summary/cnt_sensitivity_kapitza.csv")
    
    # Plot 2: Kapitza sensitivity
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(a_K_sweep, k_eff_ak, color='#27ae60', linewidth=2.5)
    ax.axvline(x=0.1, color='#e74c3c', linestyle=':', linewidth=1.5, label='Nominal a_K = 0.1 nm')
    ax.set_xlabel('Kapitza Interfacial Radius a_K (nm)', fontsize=11, fontweight='semibold')
    ax.set_ylabel('Effective Thermal Conductivity (W/mK)', fontsize=11, fontweight='semibold')
    ax.set_title("Thermal Sensitivity to Interface Resistance (V_f = 15%)", fontsize=12, fontweight='bold', pad=10)
    ax.legend(frameon=False)
    apply_academic_style(ax)
    
    plt.tight_layout()
    fig.savefig('output/figures/png/cnt_sensitivity_kapitza.png', dpi=300, bbox_inches='tight')
    fig.savefig('output/figures/pdf/cnt_sensitivity_kapitza.pdf', bbox_inches='tight')
    print("   Saved plot to output/figures/png/cnt_sensitivity_kapitza.png")
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Sweep 3: Volume Fraction and Alignment Sensitivity
    # ----------------------------------------------------------------------
    print("3. Running loading and orientation sensitivity sweep (V_f: 0 to 30%)...")
    vf_sweep = np.linspace(0.0, 0.30, 100)
    E_rand_vf = np.zeros_like(vf_sweep)
    E_align_vf = np.zeros_like(vf_sweep)
    k_rand_vf = np.zeros_like(vf_sweep)
    
    for idx, vf in enumerate(vf_sweep):
        E_rand_vf[idx] = mech_sim.halpin_tsai(
            V_f=vf, E_f=1000.0, aspect_ratio=1000.0, orientation='random', geometry='fiber'
        )
        E_align_vf[idx] = mech_sim.halpin_tsai(
            V_f=vf, E_f=1000.0, aspect_ratio=1000.0, orientation='aligned', geometry='fiber'
        )
        k_rand_vf[idx] = therm_sim.nan_model(
            V_f=vf, k_f=2000.0, aspect_ratio=1000.0, a_k=0.1,
            k_f_transverse=20.0, geometry='fiber', radius_nm=5.0
        )
        
    df_vf = pd.DataFrame({
        'Volume_Fraction': vf_sweep,
        'Volume_Fraction_Percent': vf_sweep * 100,
        'Youngs_Modulus_Random_GPa': E_rand_vf,
        'Youngs_Modulus_Aligned_GPa': E_align_vf,
        'Thermal_Conductivity_WperMk': k_rand_vf
    })
    df_vf.to_csv('output/data/summary/cnt_sensitivity_loading.csv', index=False)
    print("   Saved data to: output/data/summary/cnt_sensitivity_loading.csv")
    
    # Plot 3: Loading and orientation sensitivity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    ax1.plot(vf_sweep * 100, E_rand_vf, color='#8e44ad', linewidth=2.5, label='Random In-plane')
    ax1.plot(vf_sweep * 100, E_align_vf, color='#2c3e50', linewidth=2, linestyle='--', label='Aligned / Unidirectional')
    ax1.set_xlabel('CNT Volume Fraction (%)', fontsize=11, fontweight='semibold')
    ax1.set_ylabel("Composite Young's Modulus (GPa)", fontsize=11, fontweight='semibold')
    ax1.set_title("Modulus vs Loading & Orientation", fontsize=12, fontweight='bold', pad=10)
    ax1.legend(frameon=False)
    apply_academic_style(ax1)
    
    ax2.plot(vf_sweep * 100, k_rand_vf, color='#16a085', linewidth=2.5)
    ax2.set_xlabel('CNT Volume Fraction (%)', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Effective Thermal Conductivity (W/mK)', fontsize=11, fontweight='semibold')
    ax2.set_title("Thermal Conductivity vs Loading", fontsize=12, fontweight='bold', pad=10)
    apply_academic_style(ax2)
    
    plt.tight_layout()
    fig.savefig('output/figures/png/cnt_sensitivity_loading.png', dpi=300, bbox_inches='tight')
    fig.savefig('output/figures/pdf/cnt_sensitivity_loading.pdf', bbox_inches='tight')
    print("   Saved plot to output/figures/png/cnt_sensitivity_loading.png")
    plt.close(fig)
    
    print("=" * 60)
    print("       CNT SENSITIVITY ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == '__main__':
    run_sensitivity_analysis()
