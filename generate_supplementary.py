import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import local modules
sys.path.append('.')
sys.path.append(r"c:\Users\aumen\Desktop\PEEK-Resubmission-Package\Code")

from mechanical_models import MechanicalSimulator
from thermal_models import ThermalSimulator

# Set seed for reproducibility
np.random.seed(42)

# Directory for supplementary figures
supp_dir = r"c:\Users\aumen\Desktop\PEEK-Resubmission-Package\supplementary"
os.makedirs(supp_dir, exist_ok=True)

# Matrix properties (nominal PEEK)
matrix_properties = {
    'young_modulus': 3.6,           # GPa
    'tensile_strength': 100,        # MPa
    'thermal_conductivity': 0.25,   # W/mK [Kurtz2019]
    'poisson_ratio': 0.40
}

# Filler properties (nominal values from main.py / Table 1)
filler_properties = {
    'Carbon Fiber': {
        'young_modulus': 230.0,
        'thermal_conductivity': 20.0,
        'k_axial': 50.0,
        'k_transverse': 5.0,
        'aspect_ratio': 20.0,
        'geometry': 'fiber',
        'radius_nm': 5000.0,
        'poisson_ratio': 0.20
    },
    'Graphene Nanoplatelets': {
        'young_modulus': 1000.0,
        'thermal_conductivity': 1003.3,
        'k_axial': 3000.0,
        'k_transverse': 5.0,
        'aspect_ratio': 1000.0,
        'geometry': 'platelet',
        'radius_nm': 10.0,
        'poisson_ratio': 0.16
    },
    'Glass Fiber': {
        'young_modulus': 72.0,
        'thermal_conductivity': 1.0,
        'k_axial': 1.0,
        'k_transverse': 1.0,
        'aspect_ratio': 15.0,
        'geometry': 'fiber',
        'radius_nm': 5000.0,
        'poisson_ratio': 0.22
    },
    'Carbon Nanotubes': {
        'young_modulus': 1000.0,
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

# Set academic style options
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'stix'
})

def apply_academic_axes(ax):
    ax.tick_params(direction='in', top=True, right=True)
    ax.grid(True, which='major', linestyle='--', alpha=0.3, color='#cbd5e1')

print("=" * 60)
print("     GENERATING MISSING SUPPLEMENTARY FIGURES (S1, S2, S3)")
print("=" * 60)

# ----------------------------------------------------------------------
# 1. Figure S1: Homogenization Addition Order Sensitivity
# ----------------------------------------------------------------------
print("\n1. Generating Figure S1 (Homogenization Addition Order Sweep)...")
volume_fractions = np.linspace(0.0, 0.30, 20)

# Ordering schemes
orders = {
    'Descending Aspect Ratio (CNT/GNP -> CF -> GF)': ['Carbon Nanotubes', 'Graphene Nanoplatelets', 'Carbon Fiber', 'Glass Fiber'],
    'Ascending Aspect Ratio (GF -> CF -> GNP/CNT)': ['Glass Fiber', 'Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes'],
    'Random Addition Order (CF -> GNP -> CNT -> GF)': ['Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes', 'Glass Fiber']
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

colors = ['#bf616a', '#5e81ac', '#d08770']
markers = ['o', 's', '^']

for (name, filler_list), color, marker in zip(orders.items(), colors, markers):
    # Mechanical
    E_vals = []
    # Thermal
    k_vals = []
    
    for Vf in volume_fractions:
        if Vf == 0:
            E_vals.append(3.6)
            k_vals.append(0.25)
        else:
            fracs = [Vf / 4.0] * 4
            E = mech_sim.sequential_halpin_tsai(Vf, filler_list, fractions=fracs)
            k = therm_sim.sequential_nan_model(Vf, filler_list, fractions=fracs, a_k=0.1)
            E_vals.append(E)
            k_vals.append(k)
            
    ax1.plot(volume_fractions * 100, E_vals, color=color, linewidth=2,
             marker=marker, markersize=4, markevery=3, label=name.split(' (')[0])
    
    ax2.semilogy(volume_fractions * 100, k_vals, color=color, linewidth=2,
                 marker=marker, markersize=4, markevery=3, label=name.split(' (')[0])

ax1.set_xlabel('Total Filler Loading (vol%)', fontweight='semibold')
ax1.set_ylabel("Composite Young's Modulus (GPa)", fontweight='semibold')
ax1.set_title('(a) Mechanical Sequence Sensitivity', loc='left', pad=10)
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 45)
apply_academic_axes(ax1)

ax2.set_xlabel('Total Filler Loading (vol%)', fontweight='semibold')
ax2.set_ylabel('Effective Thermal Conductivity (W/mK)', fontweight='semibold')
ax2.set_title('(b) Thermal Sequence Sensitivity', loc='left', pad=10)
ax2.set_xlim(0, 30)
ax2.set_ylim(0.1, 150)
apply_academic_axes(ax2)

# Place legend below the plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
fig.savefig(os.path.join(supp_dir, 'Figure_S1.png'), dpi=300, bbox_inches='tight')
print("   [SUCCESS] Saved S1 plot to: supplementary/Figure_S1.png")
plt.close(fig)

# ----------------------------------------------------------------------
# 2. Figure S2: Dirichlet Randomization Sweep
# ----------------------------------------------------------------------
print("\n2. Generating Figure S2 (Dirichlet Randomization Sweep)...")
n_samples = 1000
V_f_total = 0.30

# Generate 1,000 Dirichlet samples summing to 0.30
dirichlet_fractions = np.random.dirichlet([1.0, 1.0, 1.0, 1.0], size=n_samples) * V_f_total

E_samples = np.zeros(n_samples)
k_samples = np.zeros(n_samples)

filler_list = ['Carbon Nanotubes', 'Graphene Nanoplatelets', 'Carbon Fiber', 'Glass Fiber']

for i in range(n_samples):
    fracs = list(dirichlet_fractions[i])
    # Compute mechanical using standard descending aspect ratio order
    E_samples[i] = mech_sim.sequential_halpin_tsai(V_f_total, filler_list, fractions=fracs)
    # Compute thermal
    k_samples[i] = therm_sim.sequential_nan_model(V_f_total, filler_list, fractions=fracs, a_k=0.1)

# Nominal equal-volume fraction case
E_equal = mech_sim.sequential_halpin_tsai(V_f_total, filler_list, fractions=[0.075, 0.075, 0.075, 0.075])
k_equal = therm_sim.sequential_nan_model(V_f_total, filler_list, fractions=[0.075, 0.075, 0.075, 0.075], a_k=0.1)

# Plot S2: Scatter with marginal distributions
fig = plt.figure(figsize=(8, 6.5))
grid = fig.add_gridspec(4, 4, hspace=0.15, wspace=0.15)

# Main scatter plot
ax_main = fig.add_subplot(grid[1:, :-1])
# Top histogram (Modulus)
ax_top = fig.add_subplot(grid[0, :-1], sharex=ax_main)
# Right histogram (Thermal)
ax_right = fig.add_subplot(grid[1:, -1], sharey=ax_main)

# Scatter
ax_main.scatter(E_samples, k_samples, c='#31688e', alpha=0.4, s=15, edgecolors='none', label='Random Dirichlet Samples')
# Equal distribution case
ax_main.scatter(E_equal, k_equal, c='red', marker='*', s=150, edgecolors='black', linewidth=0.8, zorder=20, label='Equal Fraction Case (7.5% each)')

ax_main.set_xlabel("Composite Young's Modulus (GPa)", fontweight='semibold')
ax_main.set_ylabel('Effective Thermal Conductivity (W/mK)', fontweight='semibold')
ax_main.set_xlim(5, 35)
ax_main.set_yscale('log')
ax_main.set_ylim(2, 600)
apply_academic_axes(ax_main)
ax_main.legend(loc='upper left', frameon=True)

# Top hist
ax_top.hist(E_samples, bins=35, color='#5e81ac', alpha=0.7, edgecolor='black', linewidth=0.5)
ax_top.axvline(E_equal, color='red', linestyle='--', linewidth=1.2)
ax_top.axis('off')
ax_top.set_title('Dirichlet Design Space at 30 vol% Total Loading', pad=10, fontweight='bold')

# Right hist
k_bins = np.logspace(np.log10(2), np.log10(600), 35)
ax_right.hist(k_samples, bins=k_bins, orientation='horizontal', color='#8fbcbb', alpha=0.7, edgecolor='black', linewidth=0.5)
ax_right.axhline(k_equal, color='red', linestyle='--', linewidth=1.2)
ax_right.axis('off')

fig.savefig(os.path.join(supp_dir, 'Figure_S2.png'), dpi=300, bbox_inches='tight')
print("   [SUCCESS] Saved S2 plot to: supplementary/Figure_S2.png")
plt.close(fig)

# Calculate statistical metrics for verification against text
E_cv = np.std(E_samples) / np.mean(E_samples)
k_cv = np.std(k_samples) / np.mean(k_samples)
print(f"   Calculated CV values for verification: Modulus CV = {E_cv*100:.1f}%, Thermal CV = {k_cv*100:.1f}%")

# ----------------------------------------------------------------------
# 3. Figure S3: Interfacial Bonding & Kapitza Sensitivity
# ----------------------------------------------------------------------
print("\n3. Generating Figure S3 (Interfacial Bonding Sweep)...")
a_K_sweep = np.linspace(0.02, 0.50, 50)
V_f_fixed = 0.15

fig, ax = plt.subplots(figsize=(6.5, 4.5))

fillers_to_sweep = {
    'Carbon Nanotubes': {'color': '#c0392b', 'marker': '^'},
    'Graphene Nanoplatelets': {'color': '#000000', 'marker': 'o'},
    'Carbon Fiber': {'color': '#2c3e50', 'marker': 's'},
    'Glass Fiber': {'color': '#2980b9', 'marker': 'D'}
}

for name, style in fillers_to_sweep.items():
    props = filler_properties[name]
    k_eff_vals = []
    
    for a_K in a_K_sweep:
        k_eff = therm_sim.nan_model(
            V_f=V_f_fixed,
            k_f=props['k_axial'],
            aspect_ratio=props['aspect_ratio'],
            a_k=a_K,
            k_f_transverse=props['k_transverse'],
            geometry=props['geometry'],
            radius_nm=props['radius_nm']
        )
        k_eff_vals.append(k_eff)
        
    ax.plot(a_K_sweep, k_eff_vals, color=style['color'], linewidth=2.0,
            marker=style['marker'], markersize=4, markevery=5,
            markerfacecolor='white', markeredgecolor=style['color'],
            label=name)

ax.set_xlabel('Kapitza Interfacial Thermal Radius, $a_K$ (nm)', fontweight='semibold')
ax.set_ylabel('Effective Thermal Conductivity, $k_{eff}$ (W/mK)', fontweight='semibold')
ax.set_title('Thermal Conductivity vs Interfacial Boundary Resistance', pad=10, fontweight='bold')
ax.set_xlim(0.02, 0.50)
ax.set_yscale('log')
ax.set_ylim(0.1, 150)
apply_academic_axes(ax)
ax.legend(loc='upper right', frameon=True)

fig.savefig(os.path.join(supp_dir, 'Figure_S3.png'), dpi=300, bbox_inches='tight')
print("   [SUCCESS] Saved S3 plot to: supplementary/Figure_S3.png")
plt.close(fig)

print("=" * 60)
print("     ALL SUPPLEMENTARY FIGURES GENERATED SUCCESSFULLY")
print("=" * 60)
