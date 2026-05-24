"""
main.py
PEEK Composite Simulation Framework
Central orchestration script for mapping the mechanical and thermal property 
landscapes of multi-filler PEEK systems with Monte Carlo uncertainty quantification.

This framework facilitates the exploration of quaternary filler interactions 
(CF, GNP, CNT, GF) using simultaneous multi-phase homogenization models with
full uncertainty propagation. Microstructure visualizations are also generated.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mechanical_models import MechanicalSimulator
from thermal_models import ThermalSimulator
from microstructure import MicrostructureGenerator
from plotting import PlotGenerator
from data_export import DataExporter
import time


class PEEKCompositeSimulation:
    """
    Main controller class for the PEEK multi-filler simulation.
    Manages the workflow from baseline single-filler sweeps to complex 
    hybrid system mapping and data export with Monte Carlo uncertainty.
    """

    def __init__(self, random_seed=42):
        """
        Initialize simulation with given random seed for reproducibility.

        Parameters:
        -----------
        random_seed : int
            Random seed for Monte Carlo simulations
        """
        # Matrix phase: Polyetheretherketone (PEEK)
        self.matrix_properties = {
            'name': 'PEEK',
            'young_modulus': 3.6,           # GPa
            'tensile_strength': 100,        # MPa
            'thermal_conductivity': 0.25,   # W/mK
            'poisson_ratio': 0.4,
            'density': 1.32                  # g/cm³
        }

        # Reinforcement phases (Filler library) with distribution parameters for Monte Carlo
        self.filler_properties = {
            'Carbon Fiber': {
                'young_modulus': 230,           # GPa [King2018] DOI: 10.1002/pc.24250
                'tensile_strength': 3500,       # MPa [King2018]
                'thermal_conductivity': 20.0,   # (50+2*5.0)/3 = 20 W/mK (isotropic average)
                'k_axial': 50.0,                # W/mK [King2018]
                'k_transverse': 5.0,            # W/mK [King2018]
                'aspect_ratio': 20,             # [Wang2020] DOI: 10.1016/j.compositesb.2020.108175
                'geometry': 'fiber',            # 1D reinforcement
                'radius_nm': 5000.0,            # 5 μm = 5000 nm [Manuscript ¶46]
                'poisson_ratio': 0.20,          # [King2018; Wang2020]
                'color': '#2c3e50',
                'marker': 'o',
                'young_modulus_dist': {'scale': 0.08, 'low': 0.7, 'high': 1.3},
                                                # 10% CV [Manuscript ¶57]
                'aspect_ratio_dist': {'s': 0.2, 'low': 5, 'high': 50},
                                                # σ=0.2 [Manuscript ¶59]
                'k_axial_dist': {'scale': 0.10, 'low': 0.5, 'high': 1.5},
                                                # 10% CV [Manuscript ¶58]
                'k_transverse_dist': {'scale': 0.10, 'low': 0.5, 'high': 1.5}
            },
            'Graphene Nanoplatelets': {
                'young_modulus': 1000,          # GPa [King2018; Chen2016]
                'tensile_strength': 5000,       # MPa
                'thermal_conductivity': 1200.0, # (3000+2*300)/3 = 1200 W/mK (isotropic average)
                'k_axial': 3000.0,              # W/mK in-plane [King2018; Chen2016]
                'k_transverse': 300.0,          # W/mK through-thickness [King2018; Chen2016]
                'aspect_ratio': 1000,           # diameter/thickness [King2018]
                'geometry': 'platelet',         # 2D reinforcement
                'radius_nm': 10.0,              # half-thickness = 10 nm [Manuscript ¶46]
                'poisson_ratio': 0.16,          # [King2018]
                'color': '#000000',
                'marker': 's',
                'young_modulus_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5},
                                                # 15% CV [Manuscript ¶57]
                'aspect_ratio_dist': {'s': 0.4, 'low': 100, 'high': 5000},
                                                # σ=0.4 [Manuscript ¶59]
                'k_axial_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5},
                                                # 15% CV [Manuscript ¶58]
                'k_transverse_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5}
            },
            'Glass Fiber': {
                'young_modulus': 72,            # GPa [Wang2020]
                'tensile_strength': 2000,       # MPa
                'thermal_conductivity': 1.0,    # W/mK (isotropic) [Wang2020]
                'k_axial': 1.0,                 # W/mK [Wang2020]
                'k_transverse': 1.0,            # W/mK [Wang2020]
                'aspect_ratio': 15,             # [Wang2020]
                'geometry': 'fiber',            # 1D reinforcement
                'radius_nm': 5000.0,            # 5 μm = 5000 nm [Manuscript ¶46]
                'poisson_ratio': 0.22,          # [Wang2020]
                'color': '#2980b9',
                'marker': '^',
                'young_modulus_dist': {'scale': 0.05, 'low': 0.8, 'high': 1.2},
                                                # 10% CV [Manuscript ¶57]
                'aspect_ratio_dist': {'s': 0.15, 'low': 10, 'high': 30},
                                                # σ=0.15 [Manuscript ¶59]
                'k_axial_dist': {'scale': 0.10, 'low': 0.5, 'high': 1.5},
                'k_transverse_dist': {'scale': 0.10, 'low': 0.5, 'high': 1.5}
            },
            'Carbon Nanotubes': {
                'young_modulus': 1000,          # GPa [Chen2016; Kim2018]
                'tensile_strength': 10000,      # MPa
                'thermal_conductivity': 680.0,   # (2000+2*20)/3 = 680 W/mK (isotropic average)
                'k_axial': 2000.0,              # W/mK axial [Chen2016]
                'k_transverse': 20.0,           # W/mK transverse [Chen2016; Kim2018]
                'aspect_ratio': 1000,           # [Chen2016]
                'geometry': 'fiber',            # 1D reinforcement
                'radius_nm': 5.0,               # 5 nm [Manuscript ¶46]
                'poisson_ratio': 0.20,          # [Kim2018; Chen2016]
                'color': '#c0392b',
                'marker': 'D',
                'young_modulus_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5},
                                                # 15% CV [Manuscript ¶57]
                'aspect_ratio_dist': {'s': 0.4, 'low': 100, 'high': 5000},
                                                # σ=0.4 [Manuscript ¶59]
                'k_axial_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5},
                                                # 15% CV [Manuscript ¶58]
                'k_transverse_dist': {'scale': 0.15, 'low': 0.5, 'high': 1.5}
            }
        }

        # Monte Carlo parameters
        self.random_seed = random_seed
        self.n_mc_samples = 10000  # Number of Monte Carlo samples

        # Sweep parameter: Volume Fraction (phi) from 0 to 30%
        self.volume_fractions = np.linspace(0, 0.3, 20)

        # Defined permutations for hybrid systems
        self.hybrid_combinations = [
            ('Carbon Fiber', 'Graphene Nanoplatelets'),
            ('Carbon Fiber', 'Carbon Nanotubes'),
            ('Graphene Nanoplatelets', 'Carbon Nanotubes'),
            ('Graphene Nanoplatelets', 'Glass Fiber'),
            ('Carbon Fiber', 'Glass Fiber'),
            ('Carbon Nanotubes', 'Glass Fiber'),
            ('Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes'),
            ('Graphene Nanoplatelets', 'Carbon Nanotubes', 'Glass Fiber'),
            ('Carbon Fiber', 'Graphene Nanoplatelets', 'Glass Fiber'),
            ('Carbon Fiber', 'Carbon Nanotubes', 'Glass Fiber'),
            ('Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes', 'Glass Fiber')
        ]

        # Module instantiation with random seed for reproducibility
        self.mech_sim = MechanicalSimulator(
            self.matrix_properties,
            self.filler_properties,
            random_state=self.random_seed
        )
        self.therm_sim = ThermalSimulator(
            self.matrix_properties,
            self.filler_properties,
            random_state=self.random_seed
        )
        self.micro_gen = MicrostructureGenerator()
        self.plotter = PlotGenerator()
        self.exporter = DataExporter()

        # Result containers
        self.mech_results = None
        self.therm_results = None
        self.mech_hybrid_results = None
        self.therm_hybrid_results = None
        self.figures = {}

        # Timing
        self.simulation_time = 0

    def run_convergence_tests(self):
        """
        Run convergence tests to verify that 10,000 samples is sufficient.
        This addresses reviewer requirement for convergence testing.
        """
        print("\n" + "-" * 30)
        print("CONVERGENCE TESTS")
        print("-" * 30)

        # Test at 15% volume fraction for one representative filler
        test_vf = 0.15
        test_filler = 'Carbon Fiber'

        print(f"  Testing convergence for {test_filler} at {test_vf*100:.0f}% volume fraction...")

        # Mechanical convergence
        mech_convergence = self.mech_sim.convergence_test(
            test_vf, test_filler,
            sample_sizes=[100, 500, 1000, 2500, 5000, 7500, 10000],
            n_repeats=5
        )

        # Thermal convergence
        therm_convergence = self.therm_sim.convergence_test(
            test_vf, test_filler,
            sample_sizes=[100, 500, 1000, 2500, 5000, 7500, 10000],
            n_repeats=5
        )

        # Print summary
        print("\n  Convergence Results:")
        print(f"    Mechanical - CV at 5000 samples: {mech_convergence['mean_stability'][4]['cv']*100:.3f}%")
        print(f"    Mechanical - CV at 7500 samples: {mech_convergence['mean_stability'][5]['cv']*100:.3f}%")
        print(f"    Mechanical - CV at 10000 samples: {mech_convergence['mean_stability'][6]['cv']*100:.3f}%")
        print(f"    Thermal - CV at 5000 samples: {therm_convergence['mean_stability'][4]['cv']*100:.3f}%")
        print(f"    Thermal - CV at 7500 samples: {therm_convergence['mean_stability'][5]['cv']*100:.3f}%")
        print(f"    Thermal - CV at 10000 samples: {therm_convergence['mean_stability'][6]['cv']*100:.3f}%")

        # Verify stability (should be < 2% variation)
        if (mech_convergence['mean_stability'][5]['cv'] < 0.02 and
            therm_convergence['mean_stability'][5]['cv'] < 0.02):
            print("\n  ✓ Convergence verified: 10,000 samples is sufficient (CV < 2%)")
        else:
            print("\n  ⚠ Warning: Consider increasing sample size for better convergence")

        return mech_convergence, therm_convergence

    def run_single_filler_simulation(self):
        """
        Generates baseline curves for each individual filler type
        with Monte Carlo uncertainty quantification.
        """
        print("\n" + "-" * 30)
        print("BASELINE SINGLE-FILLER SWEEPS (Monte Carlo)")
        print(f"  Samples per point: {self.n_mc_samples}")
        print("-" * 30)

        start_time = time.time()

        # Mechanical properties with Monte Carlo
        print("\n  Mechanical properties:")
        self.mech_results = self.mech_sim.sweep_volume_fraction_monte_carlo(
            self.volume_fractions,
            n_samples=self.n_mc_samples,
            show_progress=True
        )

        # Thermal properties with Monte Carlo
        print("\n  Thermal properties:")
        self.therm_results = self.therm_sim.sweep_volume_fraction_monte_carlo(
            self.volume_fractions,
            n_samples=self.n_mc_samples,
            show_progress=True
        )

        elapsed = time.time() - start_time
        print(f"\n  Single-filler simulation completed in {elapsed/60:.2f} minutes")

    def run_hybrid_simulation(self):
        """
        Maps hybrid property space using sequential homogenization
        with Monte Carlo uncertainty quantification.
        """
        print("\n" + "-" * 30)
        print("HYBRID SYSTEM SIMULATIONS (Monte Carlo)")
        print(f"  Samples per point: {self.n_mc_samples}")
        print(f"  Combinations: {len(self.hybrid_combinations)}")
        print("-" * 30)

        start_time = time.time()

        # Mechanical hybrid properties with Monte Carlo
        print("\n  Mechanical hybrid properties:")
        self.mech_hybrid_results = self.mech_sim.sweep_hybrid_combinations_monte_carlo(
            self.volume_fractions,
            self.hybrid_combinations,
            n_samples=self.n_mc_samples,
            show_progress=True
        )

        # Thermal hybrid properties with Monte Carlo
        print("\n  Thermal hybrid properties:")
        self.therm_hybrid_results = self.therm_sim.sweep_hybrid_combinations_monte_carlo(
            self.volume_fractions,
            self.hybrid_combinations,
            n_samples=self.n_mc_samples,
            show_progress=True
        )

        elapsed = time.time() - start_time
        print(f"\n  Hybrid simulation completed in {elapsed/60:.2f} minutes")

        self.simulation_time = elapsed

    def generate_outputs(self):
        """
        Handles RVE generation, plotting, and high-fidelity data export.
        Microstructure figures are merged with the main plots to ensure they are saved.
        """
        print("\n" + "-" * 30)
        print("GENERATING VISUALIZATIONS & EXPORTS")
        print("-" * 30)

        # 1. RVE Microstructures
        print("\n  Generating microstructures...")
        self.micro_gen.set_seed(42)
        micro_figs = {
            'microstructure_comparison': self.micro_gen.create_rve_comparison(
                self.filler_properties
            ),
            'hybrid_microstructure': self.micro_gen.create_hybrid_rve(
                ['Carbon Fiber', 'Graphene Nanoplatelets', 'Carbon Nanotubes'],
                [0.05, 0.05, 0.05],
                total_vf=0.15
            )
        }

        # 2. Scientific Plots (using Monte Carlo results with confidence intervals)
        print("\n  Creating publication plots...")

        # Convert Monte Carlo results to format expected by plotter
        mech_deterministic = self._extract_deterministic_mech()
        therm_deterministic = self._extract_deterministic_therm()

        # Note: The plotter may use the microstructure figure if needed (here we pass the comparison figure)
        plot_figs = self.plotter.create_all_plots(
            self.volume_fractions,
            mech_deterministic,
            therm_deterministic,
            self.filler_properties,
            micro_figs.get('microstructure_comparison'),  # optional, used by some plotter methods
            self.mech_hybrid_results,
            self.therm_hybrid_results
        )

        # Merge microstructure and plot figures (micro_figs + plot_figs)
        # If there are duplicate keys, plot_figs will overwrite; that's acceptable.
        self.figures = {**micro_figs, **plot_figs}

        # 3. CSV & Publication Export
        print("\n  Exporting data and figures...")
        self.exporter.export_all(
            self.volume_fractions,
            self.mech_results,      # Monte Carlo results with CIs
            self.therm_results,      # Monte Carlo results with CIs
            self.figures,
            self.mech_hybrid_results,
            self.therm_hybrid_results
        )

    def _extract_deterministic_mech(self):
        """
        Extract deterministic (mean) values from Monte Carlo results
        for compatibility with plotting functions.
        """
        mech_det = {}
        for filler, results in self.mech_results.items():
            if isinstance(results, dict) and 'young_modulus' in results:
                mech_det[filler] = {
                    'young_modulus': {
                        'Halpin-Tsai': results['young_modulus']['Halpin-Tsai']['mean'],
                        'ROM': None,
                        'Mori-Tanaka': None  # Deprecated: see mori_tanaka() docstring
                    },
                    'tensile_strength': np.zeros_like(self.volume_fractions),
                    'enhancement_factor': results['enhancement_factor']
                }
        return mech_det

    def _extract_deterministic_therm(self):
        """
        Extract deterministic (mean) values from Monte Carlo results
        for compatibility with plotting functions.
        """
        therm_det = {}
        for filler, results in self.therm_results.items():
            if '_all_models' not in filler:
                therm_det[filler] = {
                    'values': results['mean'],
                    'ci_lower': results['ci_lower'],
                    'ci_upper': results['ci_upper']
                }
            else:
                therm_det[filler] = results  # Keep all_models as is
        return therm_det

    def print_final_summary(self):
        """
        Console output of key property targets with confidence intervals.
        """
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS (@ 30% Loading with 95% CI)")
        print("=" * 70)

        # Single filler summary
        print("\nSingle Fillers:")
        print("-" * 60)
        print(f"{'Filler':<25} {'E (GPa)':<25} {'k (W/mK)':<25}")
        print("-" * 60)

        for filler in self.filler_properties.keys():
            if filler in self.mech_results and filler in self.therm_results:
                E_mean = self.mech_results[filler]['young_modulus']['Halpin-Tsai']['mean'][-1]
                E_low = self.mech_results[filler]['young_modulus']['Halpin-Tsai']['ci_lower'][-1]
                E_high = self.mech_results[filler]['young_modulus']['Halpin-Tsai']['ci_upper'][-1]

                k_mean = self.therm_results[filler]['mean'][-1]
                k_low = self.therm_results[filler]['ci_lower'][-1]
                k_high = self.therm_results[filler]['ci_upper'][-1]

                print(f"{filler:<25} {E_mean:6.2f} ({E_low:5.1f}-{E_high:5.1f})  "
                      f"{k_mean:8.3f} ({k_low:6.3f}-{k_high:6.3f})")

        # Hybrid summary
        if self.mech_hybrid_results and self.therm_hybrid_results:
            print("\nTop Hybrid Performers (at 30% total Vf):")
            print("-" * 70)
            print(f"{'Combination':<40} {'E (GPa)':<25} {'k (W/mK)':<25}")
            print("-" * 70)

            # Collect and sort by modulus
            hybrids = []
            for combo in self.mech_hybrid_results.keys():
                if combo in self.therm_hybrid_results:
                    E_mean = self.mech_hybrid_results[combo]['mean'][-1]
                    E_low = self.mech_hybrid_results[combo]['ci_lower'][-1]
                    E_high = self.mech_hybrid_results[combo]['ci_upper'][-1]

                    k_mean = self.therm_hybrid_results[combo]['mean'][-1]
                    k_low = self.therm_hybrid_results[combo]['ci_lower'][-1]
                    k_high = self.therm_hybrid_results[combo]['ci_upper'][-1]

                    hybrids.append((combo, E_mean, E_low, E_high, k_mean, k_low, k_high))

            # Sort by modulus descending
            hybrids.sort(key=lambda x: x[1], reverse=True)

            for h in hybrids[:8]:  # Show top 8
                print(f"{h[0]:<40} {h[1]:6.2f} ({h[2]:5.1f}-{h[3]:5.1f})  "
                      f"{h[4]:8.3f} ({h[5]:6.3f}-{h[6]:6.3f})")

        # Simulation info
        print("\n" + "-" * 70)
        print(f"Monte Carlo samples per point: {self.n_mc_samples}")
        print(f"Random seed: {self.random_seed}")
        print(f"Total simulation time: {self.simulation_time/60:.2f} minutes")
        print("=" * 70)

    def execute(self):
        """
        Workflow orchestrator.
        """
        print("\n" + "=" * 70)
        print("PEEK COMPOSITE SIMULATION FRAMEWORK")
        print("Systematic parametric mapping with Monte Carlo uncertainty")
        print("=" * 70)
        print(f"Random seed: {self.random_seed}")
        print(f"Monte Carlo samples: {self.n_mc_samples}")
        print(f"Volume fractions: 0-30% in {len(self.volume_fractions)} steps")
        print(f"Hybrid combinations: {len(self.hybrid_combinations)}")

        total_start = time.time()

        # Run convergence tests first (optional, can be skipped for speed)
        # self.run_convergence_tests()

        # Main simulations
        self.run_single_filler_simulation()
        self.run_hybrid_simulation()

        # Generate outputs
        self.generate_outputs()

        # Print summary
        self.print_final_summary()

        total_elapsed = time.time() - total_start
        print(f"\nFramework execution completed in {total_elapsed/60:.2f} minutes.")
        print("\nResults saved to: ./output/")
        print("=" * 70)


if __name__ == "__main__":
    # Run with fixed random seed for reproducibility
    sim = PEEKCompositeSimulation(random_seed=42)
    sim.execute()

    # Optional: Quick test with fewer samples for debugging
    # sim.n_mc_samples = 1000
    # sim.execute()