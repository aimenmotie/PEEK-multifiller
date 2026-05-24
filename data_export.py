"""
data_export.py
Data export module for saving simulation results and figures.
Handles CSV export, figure saving in multiple formats, and README generation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json


class DataExporter:
    """
    Handles export of simulation data and figures in publication-ready formats.
    Supports single-filler and multi-filler results, both deterministic and Monte Carlo.
    """
    
    def __init__(self, output_dir='output'):
        """
        Parameters:
        -----------
        output_dir : str
            Base output directory name
        """
        self.output_dir = output_dir
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create organized directory structure for outputs"""
        directories = [
            self.output_dir,
            f'{self.output_dir}/figures',
            f'{self.output_dir}/data',
            f'{self.output_dir}/figures/png',
            f'{self.output_dir}/figures/pdf',
            f'{self.output_dir}/figures/svg',
            f'{self.output_dir}/data/single_filler',
            f'{self.output_dir}/data/hybrid_filler',
            f'{self.output_dir}/data/summary'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  Created directory: {directory}")
    
    def export_figure(self, fig, filename, dpi=300):
        """
        Export figure in multiple formats for publication
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            Figure to export
        filename : str
            Base filename (without extension)
        dpi : int
            Resolution for raster formats
        """
        # Clean filename (remove spaces and special characters)
        clean_filename = filename.replace(' ', '_').replace('+', '_').replace('/', '_')
        
        # Save as PNG (raster, high resolution)
        png_path = f'{self.output_dir}/figures/png/{clean_filename}.png'
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"    Saved PNG: {png_path}")
        
        # Save as PDF (vector, for publications)
        pdf_path = f'{self.output_dir}/figures/pdf/{clean_filename}.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white', 
                   edgecolor='none')
        print(f"    Saved PDF: {pdf_path}")
        
        # Save as SVG (editable vector)
        svg_path = f'{self.output_dir}/figures/svg/{clean_filename}.svg'
        fig.savefig(svg_path, bbox_inches='tight', facecolor='white',
                   edgecolor='none')
        print(f"    Saved SVG: {svg_path}")
        
        # Close figure to free memory
        plt.close(fig)
    
    def export_data_csv(self, volume_fractions, mech_results, therm_results):
        """
        Export single-filler simulation data to CSV files.
        Handles both deterministic and Monte Carlo structures.
        
        Parameters:
        -----------
        volume_fractions : array
            Volume fractions simulated
        mech_results : dict
            Mechanical properties results (from Monte Carlo or deterministic)
        therm_results : dict
            Thermal properties results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export mechanical properties for each filler
        for filler_name, results in mech_results.items():
            # Create DataFrame for Young's modulus
            df_young = pd.DataFrame({
                'Volume_Fraction': volume_fractions,
                'Volume_Fraction_Percent': volume_fractions * 100
            })
            
            # Handle different structures
            if 'young_modulus' in results:
                ym = results['young_modulus']
                # Monte Carlo format: ym['Halpin-Tsai'] is a dict with 'mean', 'ci_lower', ...
                if 'Halpin-Tsai' in ym and isinstance(ym['Halpin-Tsai'], dict):
                    ht = ym['Halpin-Tsai']
                    if 'mean' in ht:
                        df_young['Young_Modulus_HT_mean_GPa'] = ht['mean']
                        df_young['Young_Modulus_HT_ci_lower_GPa'] = ht['ci_lower']
                        df_young['Young_Modulus_HT_ci_upper_GPa'] = ht['ci_upper']
                    else:
                        # deterministic: might be array
                        df_young['Young_Modulus_HT_GPa'] = ht
                elif 'Halpin-Tsai' in ym:
                    df_young['Young_Modulus_HT_GPa'] = ym['Halpin-Tsai']
                
                # Add other models if present
                for model in ['ROM', 'Mori-Tanaka']:
                    if model in ym:
                        df_young[f'Young_Modulus_{model}_GPa'] = ym[model]
            
            # Add enhancement factor if present
            if 'enhancement_factor' in results:
                df_young['Enhancement_Factor'] = results['enhancement_factor']
            
            # Add tensile strength if present (Monte Carlo doesn't have it)
            if 'tensile_strength' in results:
                df_young['Tensile_Strength_MPa'] = results['tensile_strength']
            
            # Save to CSV
            clean_name = filler_name.replace(' ', '_')
            filename = f'{self.output_dir}/data/single_filler/mechanical_{clean_name}_{timestamp}.csv'
            df_young.to_csv(filename, index=False, float_format='%.4f')
            print(f"    Saved mechanical data: {filename}")
        
        # Export thermal properties summary
        thermal_data = {
            'Volume_Fraction': volume_fractions,
            'Volume_Fraction_Percent': volume_fractions * 100
        }
        
        for filler_name, values in therm_results.items():
            if 'all_models' not in filler_name:
                clean_name = filler_name.replace(' ', '_')
                # Extract values: could be dict with 'mean' or just array
                if isinstance(values, dict) and 'mean' in values:
                    thermal_data[f'Thermal_Conductivity_{clean_name}_mean_WperMk'] = values['mean']
                    thermal_data[f'Thermal_Conductivity_{clean_name}_ci_lower_WperMk'] = values['ci_lower']
                    thermal_data[f'Thermal_Conductivity_{clean_name}_ci_upper_WperMk'] = values['ci_upper']
                else:
                    # deterministic: array
                    thermal_data[f'Thermal_Conductivity_{clean_name}_WperMk'] = values
        
        df_thermal = pd.DataFrame(thermal_data)
        filename = f'{self.output_dir}/data/single_filler/thermal_properties_{timestamp}.csv'
        df_thermal.to_csv(filename, index=False, float_format='%.4f')
        print(f"    Saved thermal data: {filename}")
        
        # Export detailed thermal models for each filler (if present)
        for filler_name, results in therm_results.items():
            if 'all_models' in filler_name:
                base_name = filler_name.replace('_all_models', '').replace(' ', '_')
                df_detailed = pd.DataFrame({
                    'Volume_Fraction': volume_fractions,
                    'Volume_Fraction_Percent': volume_fractions * 100
                })
                
                for model_name, vals in results.items():
                    # vals could be array or dict
                    if isinstance(vals, dict) and 'mean' in vals:
                        df_detailed[f'Thermal_{model_name}_mean_WperMk'] = vals['mean']
                    else:
                        df_detailed[f'Thermal_{model_name}_WperMk'] = vals
                
                filename = f'{self.output_dir}/data/single_filler/thermal_detailed_{base_name}_{timestamp}.csv'
                df_detailed.to_csv(filename, index=False, float_format='%.4f')
                print(f"    Saved detailed thermal data: {filename}")
    
    def export_hybrid_data_csv(self, volume_fractions, mech_hybrid_results, therm_hybrid_results):
        """
        Export hybrid (multi-filler) simulation data to CSV files.
        Handles Monte Carlo structures.
        
        Parameters:
        -----------
        volume_fractions : array
            Volume fractions simulated
        mech_hybrid_results : dict
            Mechanical properties for hybrid combinations (Monte Carlo dicts)
        therm_hybrid_results : dict
            Thermal properties for hybrid combinations
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export mechanical hybrid data
        mech_hybrid_data = {
            'Volume_Fraction': volume_fractions,
            'Volume_Fraction_Percent': volume_fractions * 100
        }
        
        for combo_name, data in mech_hybrid_results.items():
            clean_name = combo_name.replace(' ', '_').replace('+', '_')
            # data is a dict with 'mean', 'ci_lower', 'ci_upper'
            if isinstance(data, dict) and 'mean' in data:
                mech_hybrid_data[f'Young_Modulus_{clean_name}_mean_GPa'] = data['mean']
                mech_hybrid_data[f'Young_Modulus_{clean_name}_ci_lower_GPa'] = data['ci_lower']
                mech_hybrid_data[f'Young_Modulus_{clean_name}_ci_upper_GPa'] = data['ci_upper']
            else:
                # fallback to array
                mech_hybrid_data[f'Young_Modulus_{clean_name}_GPa'] = data
        
        df_mech_hybrid = pd.DataFrame(mech_hybrid_data)
        filename = f'{self.output_dir}/data/hybrid_filler/mechanical_hybrid_{timestamp}.csv'
        df_mech_hybrid.to_csv(filename, index=False, float_format='%.4f')
        print(f"    Saved hybrid mechanical data: {filename}")
        
        # Export thermal hybrid data
        therm_hybrid_data = {
            'Volume_Fraction': volume_fractions,
            'Volume_Fraction_Percent': volume_fractions * 100
        }
        
        for combo_name, data in therm_hybrid_results.items():
            clean_name = combo_name.replace(' ', '_').replace('+', '_')
            if isinstance(data, dict) and 'mean' in data:
                therm_hybrid_data[f'Thermal_Conductivity_{clean_name}_mean_WperMk'] = data['mean']
                therm_hybrid_data[f'Thermal_Conductivity_{clean_name}_ci_lower_WperMk'] = data['ci_lower']
                therm_hybrid_data[f'Thermal_Conductivity_{clean_name}_ci_upper_WperMk'] = data['ci_upper']
            else:
                therm_hybrid_data[f'Thermal_Conductivity_{clean_name}_WperMk'] = data
        
        df_therm_hybrid = pd.DataFrame(therm_hybrid_data)
        filename = f'{self.output_dir}/data/hybrid_filler/thermal_hybrid_{timestamp}.csv'
        df_therm_hybrid.to_csv(filename, index=False, float_format='%.4f')
        print(f"    Saved hybrid thermal data: {filename}")
        
        # Create a combined summary table for publication
        self.create_hybrid_summary_table(volume_fractions, mech_hybrid_results, 
                                         therm_hybrid_results, timestamp)
    
    def create_hybrid_summary_table(self, volume_fractions, mech_hybrid_results, 
                                    therm_hybrid_results, timestamp):
        """
        Create a summary table of hybrid properties at key volume fractions
        
        Parameters:
        -----------
        volume_fractions : array
            Volume fractions simulated
        mech_hybrid_results : dict
            Mechanical properties for hybrid combinations
        therm_hybrid_results : dict
            Thermal properties for hybrid combinations
        timestamp : str
            Timestamp for filename
        """
        # Key volume fractions to summarize (5%, 10%, 15%, 20%, 25%, 30%)
        key_vf = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        indices = [np.argmin(np.abs(volume_fractions - vf)) for vf in key_vf]
        
        summary_data = []
        
        for combo_name in mech_hybrid_results.keys():
            mech_data = mech_hybrid_results[combo_name]
            therm_data = therm_hybrid_results[combo_name]
            
            for i, vf_idx in enumerate(indices):
                # Extract mean values if available, else direct array
                if isinstance(mech_data, dict) and 'mean' in mech_data:
                    E_val = mech_data['mean'][vf_idx]
                    E_low = mech_data['ci_lower'][vf_idx]
                    E_high = mech_data['ci_upper'][vf_idx]
                else:
                    E_val = mech_data[vf_idx]
                    E_low = E_high = np.nan
                
                if isinstance(therm_data, dict) and 'mean' in therm_data:
                    k_val = therm_data['mean'][vf_idx]
                    k_low = therm_data['ci_lower'][vf_idx]
                    k_high = therm_data['ci_upper'][vf_idx]
                else:
                    k_val = therm_data[vf_idx]
                    k_low = k_high = np.nan
                
                row = {
                    'Combination': combo_name,
                    'Volume_Fraction': key_vf[i],
                    'Volume_Fraction_Percent': key_vf[i] * 100,
                    'Young_Modulus_GPa': E_val,
                    'Young_Modulus_ci_lower': E_low,
                    'Young_Modulus_ci_upper': E_high,
                    'Thermal_Conductivity_WperMk': k_val,
                    'Thermal_Conductivity_ci_lower': k_low,
                    'Thermal_Conductivity_ci_upper': k_high,
                    'Enhancement_Factor': E_val / 3.6 if not np.isnan(E_val) else np.nan
                }
                summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        filename = f'{self.output_dir}/data/summary/hybrid_summary_{timestamp}.csv'
        df_summary.to_csv(filename, index=False, float_format='%.3f')
        print(f"    Saved hybrid summary table: {filename}")
        
        # Also create pivot tables for easy viewing
        pivot_mech = df_summary.pivot_table(
            values='Young_Modulus_GPa', 
            index='Combination', 
            columns='Volume_Fraction_Percent'
        )
        pivot_therm = df_summary.pivot_table(
            values='Thermal_Conductivity_WperMk', 
            index='Combination', 
            columns='Volume_Fraction_Percent'
        )
        
        # Save pivot tables
        pivot_mech.to_csv(f'{self.output_dir}/data/summary/mechanical_pivot_{timestamp}.csv', 
                         float_format='%.2f')
        pivot_therm.to_csv(f'{self.output_dir}/data/summary/thermal_pivot_{timestamp}.csv', 
                          float_format='%.3f')
    
    def export_readme(self, volume_fractions, mech_results, therm_results, 
                     mech_hybrid_results=None, therm_hybrid_results=None):
        """
        Create README file with simulation details - FIXED ENCODING ISSUE
        
        Parameters:
        -----------
        volume_fractions : array
            Volume fractions simulated
        mech_results : dict
            Mechanical properties results
        therm_results : dict
            Thermal properties results
        mech_hybrid_results : dict, optional
            Mechanical properties for hybrid combinations
        therm_hybrid_results : dict, optional
            Thermal properties for hybrid combinations
        """
        readme_path = f'{self.output_dir}/README.txt'
        
        # Use UTF-8 encoding to handle special characters
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("PEEK COMPOSITE SIMULATION RESULTS\n")
            f.write("Systematic Parametric Mapping of Multi-Filler PEEK Composites\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("SIMULATION PARAMETERS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Matrix Material: PEEK\n")
            f.write(f"Matrix Modulus: 3.6 GPa\n")
            f.write(f"Matrix Thermal Conductivity: 0.25 W/mK\n")
            f.write(f"Volume Fractions: {volume_fractions[0]*100:.1f}% to {volume_fractions[-1]*100:.1f}%\n")
            f.write(f"Number of Data Points: {len(volume_fractions)}\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("FILLER MATERIALS\n")
            f.write("=" * 70 + "\n")
            for filler_name in mech_results.keys():
                # Extract values safely
                mech = mech_results[filler_name]
                therm = therm_results.get(filler_name, None)
                
                # Get mechanical at 30%
                if 'young_modulus' in mech and 'Halpin-Tsai' in mech['young_modulus']:
                    ht = mech['young_modulus']['Halpin-Tsai']
                    if isinstance(ht, dict) and 'mean' in ht:
                        E_val = ht['mean'][-1]
                    else:
                        E_val = ht[-1]
                else:
                    E_val = np.nan
                
                # Get enhancement factor
                if 'enhancement_factor' in mech:
                    enh = mech['enhancement_factor'][-1]
                else:
                    enh = np.nan
                
                # Get thermal at 30%
                if therm is not None:
                    if isinstance(therm, dict) and 'mean' in therm:
                        k_val = therm['mean'][-1]
                    elif isinstance(therm, dict) and 'values' in therm:
                        k_val = therm['values'][-1]
                    elif isinstance(therm, (list, np.ndarray)):
                        k_val = therm[-1]
                    else:
                        k_val = np.nan
                else:
                    k_val = np.nan
                
                f.write(f"\n{filler_name}:\n")
                f.write(f"  Young's Modulus: {E_val:.2f} GPa (at 30%)\n")
                f.write(f"  Enhancement Factor: {enh:.2f}x\n")
                f.write(f"  Thermal Conductivity: {k_val:.3f} W/mK (at 30%)\n")
            
            if mech_hybrid_results:
                f.write("\n" + "=" * 70 + "\n")
                f.write("HYBRID COMBINATIONS (at 30% total volume fraction)\n")
                f.write("=" * 70 + "\n")
                f.write("\n{:<40} {:>15} {:>20}\n".format("Combination", "E (GPa)", "k (W/mK)"))
                f.write("-" * 75 + "\n")
                
                last_idx = -1
                # Sort by modulus for better presentation
                hybrids = []
                for combo in mech_hybrid_results.keys():
                    if combo in therm_hybrid_results:
                        mech_data = mech_hybrid_results[combo]
                        therm_data = therm_hybrid_results[combo]
                        
                        if isinstance(mech_data, dict) and 'mean' in mech_data:
                            e_val = mech_data['mean'][last_idx]
                        else:
                            e_val = mech_data[last_idx]
                        
                        if isinstance(therm_data, dict) and 'mean' in therm_data:
                            k_val = therm_data['mean'][last_idx]
                        else:
                            k_val = therm_data[last_idx]
                        
                        hybrids.append((combo, e_val, k_val))
                
                # Sort by modulus descending
                hybrids.sort(key=lambda x: x[1], reverse=True)
                
                for combo, e_val, k_val in hybrids:
                    f.write("{:<40} {:>15.2f} {:>20.3f}\n".format(
                        combo, e_val, k_val
                    ))
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("MODELS IMPLEMENTED\n")
            f.write("=" * 70 + "\n")
            f.write("Mechanical:\n")
            f.write("  - Rule of Mixtures (ROM)\n")
            f.write("  - Halpin-Tsai (with geometry-dependent xi)\n")
            f.write("  - Mori-Tanaka mean-field homogenization\n")
            f.write("  - Sequential Halpin-Tsai for hybrid systems\n\n")
            f.write("Thermal:\n")
            f.write("  - Maxwell-Garnett (spherical inclusions)\n")
            f.write("  - Nan's model (with interfacial thermal resistance)\n")
            f.write("  - Series/Parallel bounds\n")
            f.write("  - Sequential Nan model for hybrid systems\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("FILE STRUCTURE\n")
            f.write("=" * 70 + "\n")
            f.write("/figures/png/          - High-resolution PNG files\n")
            f.write("/figures/pdf/          - Vector PDF files for publication\n")
            f.write("/figures/svg/          - Editable SVG files\n")
            f.write("/data/single_filler/   - CSV data for individual fillers\n")
            f.write("/data/hybrid_filler/   - CSV data for hybrid combinations\n")
            f.write("/data/summary/         - Summary tables and pivot tables\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("CITATION\n")
            f.write("=" * 70 + "\n")
            f.write("If you use this data in your research, please cite:\n")
            f.write("[Author Names], 'Systematic parametric mapping of mechanical and\n")
            f.write("thermal properties in multi-filler PEEK composites', [Journal], [Year]\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("CONTACT\n")
            f.write("=" * 70 + "\n")
            f.write("For questions or collaboration, contact:\n")
            f.write("[Author Email]\n")
        
        print(f"    Saved README: {readme_path}")
    
    def export_all(self, volume_fractions, mech_results, therm_results, figures,
                  mech_hybrid_results=None, therm_hybrid_results=None):
        """
        Export all data and figures
        
        Parameters:
        -----------
        volume_fractions : array
            Volume fractions simulated
        mech_results : dict
            Mechanical properties results
        therm_results : dict
            Thermal properties results
        figures : dict
            Dictionary of matplotlib figures
        mech_hybrid_results : dict, optional
            Mechanical properties for hybrid combinations
        therm_hybrid_results : dict, optional
            Thermal properties for hybrid combinations
        """
        print("\n  Exporting figures...")
        for name, fig in figures.items():
            self.export_figure(fig, name)
        
        print("\n  Exporting single-filler data...")
        self.export_data_csv(volume_fractions, mech_results, therm_results)
        
        if mech_hybrid_results is not None and therm_hybrid_results is not None:
            print("\n  Exporting hybrid filler data...")
            self.export_hybrid_data_csv(volume_fractions, mech_hybrid_results, therm_hybrid_results)
        
        print("\n  Creating README...")
        self.export_readme(volume_fractions, mech_results, therm_results,
                          mech_hybrid_results, therm_hybrid_results)
        
        print("\n  All exports completed successfully!")
        
        # Print summary of exported files
        print("\n  " + "-" * 40)
        print("  EXPORT SUMMARY")
        print("  " + "-" * 40)
        print(f"  Figures exported: {len(figures)}")
        print(f"  Single-filler datasets: {len(mech_results)}")
        if mech_hybrid_results:
            print(f"  Hybrid combinations: {len(mech_hybrid_results)}")
        print("  " + "-" * 40)