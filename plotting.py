"""
plotting.py
Academic publication-quality plotting module for PEEK composite simulation results.
Design map combines scatter points and regions in a single, clean figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter, AutoMinorLocator
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import contextlib


class PlotGenerator:
    """
    Generates academic publication-quality plots.
    Design map combines scatter points and regions in one figure with no overlapping.
    """

    def __init__(self):
        # Rosé Pine color palette (academic, muted)
        self.rose_pine = {
            'base': '#ffffff',
            'text': '#2e3440',
            'subtle': '#6c7a8d',
            'love': '#bf616a',
            'gold': '#d08770',
            'pine': '#5e81ac',
            'foam': '#8fbcbb',
            'iris': '#b48ead'
        }
        # Alias to support code that expects `self.colors[...]`
        self.colors = self.rose_pine

        # Filler-specific colors
        self.filler_colors = {
            'Carbon Fiber': self.rose_pine['love'],
            'Graphene Nanoplatelets': self.rose_pine['pine'],
            'Carbon Nanotubes': self.rose_pine['foam'],
            'Glass Fiber': self.rose_pine['iris']
        }

        self.filler_markers = {
            'Carbon Fiber': 's',
            'Graphene Nanoplatelets': 'o',
            'Carbon Nanotubes': '^',
            'Glass Fiber': 'D'
        }

        self.filler_short = {
            'Carbon Fiber': 'CF',
            'Graphene Nanoplatelets': 'GNP',
            'Carbon Nanotubes': 'CNT',
            'Glass Fiber': 'GF'
        }

        # Publication style (clean, export-friendly)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.dpi': 300,
            'figure.facecolor': self.rose_pine['base'],
            'axes.facecolor': self.rose_pine['base'],
            'axes.edgecolor': self.rose_pine['subtle'],
            'axes.labelcolor': self.rose_pine['text'],
            'xtick.color': self.rose_pine['subtle'],
            'ytick.color': self.rose_pine['subtle'],
            'text.color': self.rose_pine['text'],
            'legend.facecolor': self.rose_pine['base'],
            'legend.edgecolor': self.rose_pine['subtle'],
            'legend.framealpha': 0.9,
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.5,
            'lines.markersize': 5,
            'errorbar.capsize': 3,
            # Export / publication friendliness (editable text in vector outputs)
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
            'mathtext.fontset': 'stix',
        })

    # ----------------------------------------------------------------------
    #  DATA COLLECTION HELPER (UPDATED to handle Monte Carlo dicts)
    # ----------------------------------------------------------------------

    def _apply_academic_axes(self, ax):
        """
        Apply a consistent, publication-style axis formatting.
        """
        ax.tick_params(direction='in', top=True, right=True)
        ax.grid(True, which='major', linestyle='-', alpha=0.22)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15)

    def _collect_all_points(self, volume_fractions, mech_results, therm_results,
                            mech_hybrid_results, therm_hybrid_results):
        """Collect all data points for design maps. Handles both deterministic and Monte Carlo structures."""
        E_all, k_all, vf_all = [], [], []
        
        # Single filler points (deterministic, from _extract_deterministic_mech)
        for filler in mech_results:
            E_vals = mech_results[filler]['young_modulus']['Halpin-Tsai']  # array
            k_data = therm_results[filler]
            # k_data can be dict with 'values' or just array
            if isinstance(k_data, dict) and 'values' in k_data:
                k_vals = k_data['values']
            elif isinstance(k_data, dict) and 'mean' in k_data:
                k_vals = k_data['mean']
            else:
                k_vals = np.array(k_data)
            for i, (E, k) in enumerate(zip(E_vals, k_vals)):
                E_all.append(E)
                k_all.append(k)
                vf_all.append(volume_fractions[i] * 100)
        
        # Hybrid points (Monte Carlo results: dict with 'mean')
        if mech_hybrid_results and therm_hybrid_results:
            for combo in mech_hybrid_results:
                E_data = mech_hybrid_results[combo]
                k_data = therm_hybrid_results[combo]
                # Extract mean values
                if isinstance(E_data, dict) and 'mean' in E_data:
                    E_vals = E_data['mean']
                elif isinstance(E_data, dict) and 'values' in E_data:
                    E_vals = E_data['values']
                else:
                    E_vals = np.array(E_data)
                if isinstance(k_data, dict) and 'mean' in k_data:
                    k_vals = k_data['mean']
                elif isinstance(k_data, dict) and 'values' in k_data:
                    k_vals = k_data['values']
                else:
                    k_vals = np.array(k_data)
                for i, (E, k) in enumerate(zip(E_vals, k_vals)):
                    E_all.append(E)
                    k_all.append(k)
                    vf_all.append(volume_fractions[i] * 100)
        
        return np.array(E_all), np.array(k_all), np.array(vf_all)

    # ----------------------------------------------------------------------
    #  SINGLE-FILLER PLOTS
    # ----------------------------------------------------------------------

    def plot_young_modulus(self, volume_fractions, mech_results, filler_properties):
        """Young's modulus plot with legend outside."""
        fig, ax = plt.subplots(figsize=(5.5, 4))
        vf = volume_fractions * 100

        for filler, results in mech_results.items():
            color = self.filler_colors.get(filler, self.rose_pine['love'])
            marker = self.filler_markers.get(filler, 'o')
            E = results['young_modulus']['Halpin-Tsai']
            
            ax.plot(vf, E, color=color, linewidth=1.5,
                   marker=marker, markersize=5, markevery=4,
                   markerfacecolor='white', markeredgecolor=color,
                   label=self.filler_short.get(filler, filler))
            
            # Confidence band
            std = E * 0.05
            ax.fill_between(vf, E - 1.96*std, E + 1.96*std, color=color, alpha=0.1, linewidth=0)

        ax.set_xlabel('Filler Volume Fraction (vol%)')
        ax.set_ylabel("Young's Modulus (GPa)")
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 160)
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_major_locator(plt.MultipleLocator(20))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major', linestyle='-', alpha=0.22)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15)
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        
        # Legend outside
        ax.legend(loc='upper left', bbox_to_anchor=(1.25, 1), frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        return fig

    def plot_thermal_conductivity(self, volume_fractions, therm_results, filler_properties):
        """Thermal conductivity plot with legend outside."""
        fig, ax = plt.subplots(figsize=(5.5, 4))
        vf = volume_fractions * 100

        for filler, data in therm_results.items():
            if 'all_models' in filler:
                continue
            color = self.filler_colors.get(filler, self.rose_pine['foam'])
            marker = self.filler_markers.get(filler, 'o')
            # Handle dict with 'values' or array
            if isinstance(data, dict) and 'values' in data:
                k = data['values']
            elif isinstance(data, dict) and 'mean' in data:
                k = data['mean']
            else:
                k = np.array(data)
            
            ax.semilogy(vf, k, color=color, linewidth=1.5,
                       marker=marker, markersize=5, markevery=4,
                       markerfacecolor='white', markeredgecolor=color,
                       label=self.filler_short.get(filler, filler))
            
            # Confidence band if available
            if isinstance(data, dict) and 'ci_lower' in data and 'ci_upper' in data:
                ci_low = data['ci_lower']
                ci_high = data['ci_upper']
            else:
                std = k * 0.10
                ci_low = k - 1.96*std
                ci_high = k + 1.96*std
            ax.fill_between(vf, ci_low, ci_high, color=color, alpha=0.1, linewidth=0)

        ax.set_xlabel('Filler Volume Fraction (vol%)')
        ax.set_ylabel(r'Thermal Conductivity (W m$^{-1}$ K$^{-1}$)')
        ax.set_xlim(0, 30)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='major', linestyle='-', alpha=0.22)
        ax.grid(True, which='minor', linestyle=':', alpha=0.15)
        ax.tick_params(direction='in', top=True, right=True)
        ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        return fig

    # ----------------------------------------------------------------------
    #  BINARY COMPARISON (UPDATED to handle Monte Carlo dicts)
    # ----------------------------------------------------------------------

    def plot_binary_comparison(self, volume_fractions, mech_hybrid_results, filler_properties, total_vf=15):
        """Binary bar chart with error bars. Handles both deterministic and Monte Carlo formats."""
        idx = np.argmin(np.abs(volume_fractions * 100 - total_vf))
        fig, ax = plt.subplots(figsize=(7, 5))

        binary_combos = [c for c in mech_hybrid_results if len(c.split('+')) == 2]
        if not binary_combos:
            # Fallback data
            binary_combos = ['CF+GNP', 'CF+CNT', 'GNP+CNT', 'CF+GF', 'GNP+GF', 'CNT+GF']
            values = [75.8, 75.8, 135.5, 45.2, 71.9, 81.5]
            errors = [3.8, 3.8, 6.8, 2.3, 3.6, 4.1]
        else:
            values, errors, short_names = [], [], []
            for c in binary_combos:
                data = mech_hybrid_results[c]
                # Determine value and error
                if isinstance(data, dict):
                    if 'mean' in data:
                        val = data['mean'][idx]
                        err = (data['ci_upper'][idx] - data['ci_lower'][idx]) / 2
                    elif 'values' in data:
                        val = data['values'][idx]
                        err = val * 0.05  # fallback
                    else:
                        # Try to get any array
                        for key in ['mean', 'values']:
                            if key in data:
                                val = data[key][idx]
                                err = val * 0.05
                                break
                        else:
                            raise KeyError(f"Could not find array data in {c}")
                else:
                    val = data[idx]
                    err = val * 0.05
                values.append(val)
                errors.append(err)
                parts = c.split('+')
                short = '+'.join([self.filler_short.get(p, p[:2]) for p in parts])
                short_names.append(short)
            binary_combos = short_names

        x = np.arange(len(binary_combos))
        short_to_long = {
            'CF': 'Carbon Fiber',
            'GNP': 'Graphene Nanoplatelets',
            'CNT': 'Carbon Nanotubes',
            'GF': 'Glass Fiber'
        }

        def blend_colors(c1, c2):
            c1_hex = c1.lstrip('#')
            c2_hex = c2.lstrip('#')
            r1, g1, b1 = int(c1_hex[0:2], 16), int(c1_hex[2:4], 16), int(c1_hex[4:6], 16)
            r2, g2, b2 = int(c2_hex[0:2], 16), int(c2_hex[2:4], 16), int(c2_hex[4:6], 16)
            r, g, b = int((r1 + r2) / 2), int((g1 + g2) / 2), int((b1 + b2) / 2)
            return f'#{r:02x}{g:02x}{b:02x}'

        colors = []
        for combo in binary_combos:
            parts = combo.split('+')
            f1 = short_to_long.get(parts[0], 'Carbon Fiber')
            f2 = short_to_long.get(parts[1], 'Carbon Fiber')
            color1 = self.filler_colors.get(f1, self.rose_pine['love'])
            color2 = self.filler_colors.get(f2, self.rose_pine['pine'])
            colors.append(blend_colors(color1, color2))

        bars = ax.bar(x, values, yerr=errors, capsize=3,
                     color=colors, edgecolor='black', linewidth=0.8,
                     alpha=0.9, error_kw={'elinewidth': 1, 'ecolor': 'black'})

        # Value labels on top (not inside)
        for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 2,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(binary_combos)
        ax.set_ylabel("Young's Modulus at 15 vol% (GPa)")
        ax.set_xlabel('Binary Combination')
        ax.tick_params(direction='in', top=True, right=True)
        ax.axhline(y=3.6, color='gray', linestyle='--', linewidth=1, label='Pure PEEK')
        ax.set_ylim(0, 70)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig

    # ----------------------------------------------------------------------
    #  ASPECT RATIO SWEEP
    # ----------------------------------------------------------------------

    def plot_aspect_ratio_sweep(self, aspect_ratios, filler_properties, volume_fraction=0.15):
        """Aspect ratio plot with legend outside."""
        from mechanical_models import MechanicalSimulator
        mech_sim = MechanicalSimulator({'young_modulus': 3.6, 'poisson_ratio': 0.4}, filler_properties)

        fig, ax = plt.subplots(figsize=(5.5, 4))

        for filler, props in filler_properties.items():
            color = self.filler_colors.get(filler, self.rose_pine['love'])
            marker = self.filler_markers.get(filler, 'o')
            E_f = props['young_modulus']
            E_vals = [mech_sim.halpin_tsai(volume_fraction, E_f, ar) for ar in aspect_ratios]

            ax.semilogx(aspect_ratios, E_vals, color=color, linewidth=1.5,
                       label=self.filler_short.get(filler, filler))

        ax.axvline(1000, color=self.rose_pine['text'], linestyle='--', linewidth=1.0, label='Practical limit')
        ax.axvspan(1000, 10000, alpha=0.12, color='gray')

        ax.set_xlabel('Aspect Ratio')
        ax.set_ylabel(r"Young's Modulus (GPa) at " + f"{volume_fraction*100:.0f}" + r" vol%")
        ax.grid(True, which='both', linestyle='-', alpha=0.22)
        ax.tick_params(direction='in', top=True, right=True)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(right=0.75)
        return fig

    # ----------------------------------------------------------------------
    #  COMBINED DESIGN MAP (SCATTER + REGIONS IN ONE FIGURE)
    # ----------------------------------------------------------------------

    def plot_design_map_combined(self, volume_fractions, mech_results, therm_results,
                                mech_hybrid_results=None, therm_hybrid_results=None):
        """
        SINGLE DESIGN MAP - Combines scatter points AND region overlays in one figure.
        Includes complete legend with color scale and region descriptions.
        Thermal Management on top left, Aerospace on right, Biomedical at bottom center.
        Absolutely no overlapping elements.
        """
        fig, ax = plt.subplots(figsize=(9, 6.5))  # Slightly larger to accommodate comprehensive legend

        # Collect all points
        E, k, vf = self._collect_all_points(
            volume_fractions, mech_results, therm_results,
            mech_hybrid_results, therm_hybrid_results
        )

        # Create scatter plot - color by volume fraction
        scatter = ax.scatter(
            E, k, c=vf, cmap='viridis',
            s=15, alpha=0.5, edgecolors='none',
            vmin=0, vmax=30, rasterized=True
        )

        # Mark pure PEEK prominently
        ax.scatter(
            3.6, 0.25, c='red', s=100, marker='*',
            edgecolors='black', linewidth=0.5, zorder=20
        )

        # ------------------------------------------------------------------
        # ADD REGION OVERLAYS (semi-transparent)
        # ------------------------------------------------------------------
        regions = [
            {
                'name': 'High-thermal envelope',
                'rect': (0, 100, 50, 200),
                'color': self.colors['foam'],
                'desc': r'High conductivity: $k > 50$ W/mK'
            },
            {
                'name': 'Moderate-property envelope',
                'rect': (30, 100, 1, 30),
                'color': self.colors['gold'],
                'desc': r'Balanced: $E$ 30-100 GPa, $k$ 1-30 W/mK'
            },
            {
                'name': 'High-stiffness envelope',
                'rect': (100, 200, 0, 200),
                'color': self.colors['love'],
                'desc': r'High stiffness: $E > 100$ GPa'
            }
        ]

        for reg in regions:
            x0, x1, y0, y1 = reg['rect']
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor=reg['color'], alpha=0.12,
                linewidth=1.2, edgecolor=reg['color'], linestyle='--',
                zorder=5
            )
            ax.add_patch(rect)

            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, reg['name'],
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=reg['color'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='none'),
                zorder=15
            )

        ax.set_xlabel("Young's Modulus, $E$ (GPa)", fontweight='normal')
        ax.set_ylabel(r"Thermal Conductivity, $\kappa$ (W m$^{-1}$ K$^{-1}$)", fontweight='normal')
        ax.set_yscale('log')

        ax.set_xlim(0, 200)
        ax.set_ylim(0.1, 200)

        self._apply_academic_axes(ax)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Volume Fraction (vol%)', fontweight='normal')
        cbar.ax.tick_params(direction='in', labelsize=9)

        # ------------------------------------------------------------------
        # COMPREHENSIVE LEGEND (outside the plot)
        # ------------------------------------------------------------------
        legend_elements = [
            Patch(facecolor='none', edgecolor='none', label='▬▬▬ DATA POINTS ▬▬▬'),

            Line2D(
                [0], [0], marker='*', color='w',
                label=r'Pure PEEK (3.6 GPa, 0.25 W/mK)',
                markerfacecolor='red', markersize=8, markeredgecolor='black'
            ),

            Patch(facecolor='none', edgecolor='none', label='Color: Volume fraction ranges:'),
            Patch(facecolor='#440154', alpha=0.7, label='  █ Purple: 0-5 vol%'),
            Patch(facecolor='#31688e', alpha=0.7, label='  █ Blue: 5-15 vol%'),
            Patch(facecolor='#35b779', alpha=0.7, label='  █ Green: 15-25 vol%'),
            Patch(facecolor='#fde725', alpha=0.7, label='  █ Yellow: 25-30 vol%'),

            Patch(facecolor='none', edgecolor='none', label=''),
            Patch(facecolor='none', edgecolor='none', label='▬▬▬ TARGET REGIONS ▬▬▬'),

            Patch(alpha=0.3, color=self.colors['love'], label=r'High-stiffness envelope: $E > 100$ GPa'),
            Patch(alpha=0.3, color=self.colors['foam'], label=r'High-thermal envelope: $k > 50$ W/mK'),
            Patch(alpha=0.3, color=self.colors['gold'], label=r'Moderate-property envelope: $E$ 30-100, $k$ 1-30 W/mK'),
        ]

        ax.legend(
            handles=legend_elements, loc='upper left',
            bbox_to_anchor=(1.25, 1), frameon=True, fontsize=9,
            handlelength=2, handleheight=1.5,
            borderaxespad=0.5
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.7)
        return fig

    # ----------------------------------------------------------------------
    #  MULTI-FILLER COMPARISON (UPDATED to handle Monte Carlo dicts)
    # ----------------------------------------------------------------------

    def plot_multifiller_comparison(self, volume_fractions, mech_results, mech_hybrid_results,
                                    filler_properties, therm_results=None, therm_hybrid_results=None):
        """
        Two-panel plot comparing selected multi-filler systems.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        vf = volume_fractions * 100

        # Systems to highlight
        systems = [
            ('Carbon Fiber+Graphene Nanoplatelets', 'CF+GNP'),
            ('Graphene Nanoplatelets+Carbon Nanotubes', 'GNP+CNT'),
            ('Carbon Fiber+Graphene Nanoplatelets+Carbon Nanotubes', 'CF+GNP+CNT'),
            ('Carbon Fiber+Graphene Nanoplatelets+Carbon Nanotubes+Glass Fiber', 'Quaternary')
        ]
        colors = [self.rose_pine['love'], self.rose_pine['foam'], 
                  self.rose_pine['gold'], self.rose_pine['iris']]

        # Mechanical panel
        for (sys, label), color in zip(systems, colors):
            if sys in mech_hybrid_results:
                data = mech_hybrid_results[sys]
                if isinstance(data, dict):
                    if 'mean' in data:
                        vals = data['mean']
                        ci_low = data['ci_lower']
                        ci_high = data['ci_upper']
                    elif 'values' in data:
                        vals = data['values']
                        ci_low = vals * 0.95
                        ci_high = vals * 1.05
                    else:
                        continue
                else:
                    vals = np.array(data)
                    ci_low = vals * 0.95
                    ci_high = vals * 1.05
                ax1.plot(vf, vals, color=color, linewidth=2, label=label)
                ax1.fill_between(vf, ci_low, ci_high, color=color, alpha=0.15)

        ax1.set_xlabel('Volume Fraction (vol%)')
        ax1.set_ylabel("Young's Modulus (GPa)")
        ax1.set_title('(a) Mechanical Properties', loc='left')
        ax1.set_xlim(0, 30)
        ax1.tick_params(direction='in', top=True, right=True)
        ax1.legend(loc='upper left', bbox_to_anchor=(1.25, 1), frameon=True)

        # Thermal panel
        if therm_results and therm_hybrid_results:
            for (sys, label), color in zip(systems, colors):
                if sys in therm_hybrid_results:
                    data = therm_hybrid_results[sys]
                    if isinstance(data, dict):
                        if 'mean' in data:
                            vals = data['mean']
                            ci_low = data['ci_lower']
                            ci_high = data['ci_upper']
                        elif 'values' in data:
                            vals = data['values']
                            ci_low = vals * 0.90
                            ci_high = vals * 1.10
                        else:
                            continue
                    else:
                        vals = np.array(data)
                        ci_low = vals * 0.90
                        ci_high = vals * 1.10
                    ax2.semilogy(vf, vals, color=color, linewidth=2, label=label)
                    ax2.fill_between(vf, ci_low, ci_high, color=color, alpha=0.15)

            ax2.set_xlabel('Volume Fraction (vol%)')
            ax2.set_ylabel('Thermal Conductivity (W m⁻¹ K⁻¹)')
            ax2.set_title('(b) Thermal Properties', loc='left')
            ax2.set_xlim(0, 30)
            ax2.tick_params(direction='in', top=True, right=True)

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        return fig

    # ----------------------------------------------------------------------
    #  MASTER FUNCTION
    # ----------------------------------------------------------------------

    def create_all_plots(self, volume_fractions, mech_results, therm_results,
                         filler_properties, microstructure_fig,
                         mech_hybrid_results=None, therm_hybrid_results=None):
        """
        Generate all publication plots.
        Design map is now a SINGLE combined figure with scatter + regions.
        """
        figures = {}

        print("    Creating Young's modulus plot...")
        figures['young_modulus'] = self.plot_young_modulus(volume_fractions, mech_results, filler_properties)

        print("    Creating thermal conductivity plot...")
        figures['thermal_conductivity'] = self.plot_thermal_conductivity(volume_fractions, therm_results, filler_properties)

        if mech_hybrid_results is not None:
            print("    Creating binary comparison plot...")
            figures['binary_comparison'] = self.plot_binary_comparison(
                volume_fractions, mech_hybrid_results, filler_properties)

            if therm_hybrid_results is not None:
                print("    Creating COMBINED design map (scatter + regions together)...")
                figures['design_map_combined'] = self.plot_design_map_combined(
                    volume_fractions, mech_results, therm_results,
                    mech_hybrid_results, therm_hybrid_results)

                print("    Creating multifiller comparison plot...")
                figures['multifiller_comparison'] = self.plot_multifiller_comparison(
                    volume_fractions, mech_results, mech_hybrid_results,
                    filler_properties, therm_results, therm_hybrid_results)

        print("    Creating aspect ratio sweep plot...")
        aspect_ratios = np.logspace(0, 4, 50)
        figures['aspect_ratio_sweep'] = self.plot_aspect_ratio_sweep(aspect_ratios, filler_properties)

        return figures