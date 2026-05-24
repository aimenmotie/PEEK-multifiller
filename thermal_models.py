"""
thermal_models.py
Thermal conductivity prediction models for PEEK composites with Monte Carlo uncertainty quantification.

Implements multi-phase effective medium theories (EMT) for thermal transport.
For hybrid systems a volume-weighted sequential homogenization approach is used.
Uncertainty is propagated via Monte Carlo sampling of input parameters.
"""

import numpy as np
from scipy import stats
from scipy.optimize import fsolve
from typing import Dict, List, Union, Optional, Tuple, Any


class ThermalSimulator:
    """
    Simulates thermal conductivity of PEEK composites using classical EMT frameworks
    with Monte Carlo uncertainty quantification.

    Models implemented:
    - Maxwell-Garnett: Dilute limit approximation (spherical inclusions).
    - Bruggeman: Symmetric effective medium theory.
    - Nan's Model: Incorporates Kapitza interfacial thermal resistance.
    - Multi-phase sequential Nan: For hybrid networks with uncertainty propagation.

    Parameters
    ----------
    matrix_properties : dict
        Must contain key: 'thermal_conductivity' (W/mK).
    filler_properties : dict
        Nested dictionary. Each filler entry must contain:
            'thermal_conductivity' (W/mK, nominal value),
            'aspect_ratio' (nominal),
            optionally distribution parameters for uncertainty.
    random_state : int or np.random.Generator, optional
        Seed for reproducible sampling.
    """

    def __init__(self,
                 matrix_properties: Dict,
                 filler_properties: Dict,
                 random_state: Optional[int] = None):
        self.k_m = matrix_properties['thermal_conductivity']
        self.fillers = filler_properties

        # Random generator for reproducible sampling
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

        # Default distribution settings (can be overridden per filler)
        self.dist_config = {
            'thermal_conductivity': {
                'type': 'truncnorm',
                'loc': 1.0,          # Will be scaled by nominal value
                'scale': 0.1,         # 10% CV
                'low': 0.5,           # Lower bound as fraction of nominal
                'high': 1.5            # Upper bound as fraction of nominal
            },
            'aspect_ratio': {
                'type': 'lognorm',
                's': 0.3,              # Shape parameter
                'low': 1.0,             # Minimum aspect ratio
                'high': None            # No upper bound by default
            },
            'kapitza_radius': {
                'type': 'truncnorm',
                'loc': 0.1,             # Default Kapitza radius (nm)
                'scale': 0.02,           # 20% CV
                'low': 0.05,             # Lower bound (nm)
                'high': 0.2               # Upper bound (nm)
            }
        }

    # ----------------------------------------------------------------------
    #  Single‑filler models (deterministic)
    # ----------------------------------------------------------------------

    def maxwell_model(self, V_f: Union[float, np.ndarray], k_f: float) -> Union[float, np.ndarray]:
        """
        Maxwell-Garnett approximation for spherical inclusions.
        """
        numerator = k_f + 2 * self.k_m + 2 * V_f * (k_f - self.k_m)
        denominator = k_f + 2 * self.k_m - V_f * (k_f - self.k_m)

        if np.any(np.abs(denominator) < 1e-12):
            return self.parallel_model(V_f, k_f)
        return self.k_m * numerator / denominator

    def bruggeman_model(self, V_f: Union[float, np.ndarray], k_f: float) -> Union[float, np.ndarray]:
        """
        Symmetric Bruggeman model. Requires numerical solving.
        """
        def objective(k_eff, V_f, k_f):
            term1 = (1 - V_f) * (self.k_m - k_eff) / (self.k_m + 2 * k_eff)
            term2 = V_f * (k_f - k_eff) / (k_f + 2 * k_eff)
            return term1 + term2

        if isinstance(V_f, np.ndarray):
            k_eff = np.zeros_like(V_f)
            for i, vf in enumerate(V_f):
                if vf == 0:
                    k_eff[i] = self.k_m
                else:
                    k0 = self.parallel_model(vf, k_f)
                    try:
                        k_eff[i] = fsolve(objective, k0, args=(vf, k_f))[0]
                    except:
                        k_eff[i] = self.maxwell_model(vf, k_f)
            return k_eff
        else:
            if V_f == 0:
                return self.k_m
            k0 = self.parallel_model(V_f, k_f)
            try:
                return fsolve(objective, k0, args=(V_f, k_f))[0]
            except:
                return self.maxwell_model(V_f, k_f)

    def nan_model(self, V_f: Union[float, np.ndarray], k_f: float,
                  aspect_ratio: float, a_k: float = 0.1,
                  k_f_transverse: Optional[float] = None,
                  geometry: str = 'fiber',
                  radius_nm: float = 5000.0) -> Union[float, np.ndarray]:
        """
        Nan's model incorporating Kapitza interfacial thermal resistance.
        Now delegates to nan_anisotropic.nan_thermal_conductivity()
        implementing [Nan1997] Eq. 23 (DOI: 10.1063/1.365209).

        If k_f_transverse is None, falls back to isotropic (k_f_transverse = k_f)
        so legacy CF/GF code paths work unchanged.
        """
        from nan_anisotropic import nan_thermal_conductivity

        if k_f_transverse is None:
            k_f_transverse = k_f  # isotropic fallback

        if isinstance(V_f, np.ndarray):
            return np.array([self.nan_model(vf, k_f, aspect_ratio, a_k,
                                            k_f_transverse, geometry, radius_nm) for vf in V_f])

        return nan_thermal_conductivity(
            V_f=V_f, k_m=self.k_m,
            k_f_axial=k_f, k_f_transverse=k_f_transverse,
            aspect_ratio=aspect_ratio, geometry=geometry,
            radius_nm=radius_nm, a_K_nm=a_k
        )

    def series_model(self, V_f: Union[float, np.ndarray], k_f: float) -> Union[float, np.ndarray]:
        """Reuss Bound (Absolute Lower Limit)"""
        return 1 / ((1 - V_f) / self.k_m + V_f / k_f)

    def parallel_model(self, V_f: Union[float, np.ndarray], k_f: float) -> Union[float, np.ndarray]:
        """Voigt Bound (Absolute Upper Limit)"""
        return (1 - V_f) * self.k_m + V_f * k_f

    # ----------------------------------------------------------------------
    #  Multi‑filler sequential homogenisation (deterministic)
    # ----------------------------------------------------------------------
    def sequential_nan_model(self,
                              V_f_total: Union[float, np.ndarray],
                              filler_list: List[str],
                              fractions: Optional[List[float]] = None,
                              a_k: float = 0.1) -> Union[float, np.ndarray]:
        """
        Sequential Nan model for hybrid composites (deterministic, nominal values).

        Fillers are added one by one, with the composite becoming the matrix
        for the next filler addition.

        Parameters
        ----------
        V_f_total : total volume fraction (0-1)
        filler_list : names of fillers in the hybrid
        fractions : individual volume fractions (must sum to V_f_total)
        a_k : Kapitza radius (nm)

        Returns
        -------
        Composite thermal conductivity (W/mK)
        """
        if fractions is None:
            n = len(filler_list)
            fractions = [V_f_total / n] * n
        else:
            if abs(sum(fractions) - V_f_total) > 1e-12:
                raise ValueError("Sum of fractions must equal V_f_total")

        if isinstance(V_f_total, np.ndarray):
            return np.array([self.sequential_nan_model(vf, filler_list, fractions, a_k)
                           for vf in V_f_total])

        if V_f_total == 0:
            return self.k_m

        # Start with matrix
        k_current = self.k_m

        for filler_name, vf in zip(filler_list, fractions):
            if vf == 0:
                continue
            props = self.fillers[filler_name]
            k_f = props.get('k_axial', props['thermal_conductivity'])
            k_f_trans = props.get('k_transverse', k_f)
            ar = props['aspect_ratio']
            geom = props.get('geometry', 'fiber')
            rad = props.get('radius_nm', 5000.0)

            # Nan model using current matrix
            original_k_m = self.k_m
            self.k_m = k_current
            k_comp = self.nan_model(vf, k_f, ar, a_k,
                                    k_f_transverse=k_f_trans,
                                    geometry=geom, radius_nm=rad)
            self.k_m = original_k_m

            # Update matrix for next filler
            k_current = k_comp

        return k_current

    # ----------------------------------------------------------------------
    #  Uncertainty propagation helpers (Monte Carlo)
    # ----------------------------------------------------------------------
    def sample_filler_properties(self, n_samples: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Draw random samples of filler properties from specified distributions.

        Returns a nested dictionary: filler_name -> property_name -> array of size n_samples.
        """
        samples = {}
        for name, props in self.fillers.items():
            samples[name] = {}

            # Thermal conductivity (k_axial)
            k_nominal = props.get('k_axial', props['thermal_conductivity'])
            dist_k = self.dist_config['thermal_conductivity'].copy()
            if 'thermal_conductivity_dist' in props:
                dist_k.update(props['thermal_conductivity_dist'])
            if 'k_axial_dist' in props:
                dist_k.update(props['k_axial_dist'])
            samples[name]['thermal_conductivity'] = self._sample_from_dist(
                dist_k, n_samples, nominal=k_nominal
            )
            # Replicate key as k_axial
            samples[name]['k_axial'] = samples[name]['thermal_conductivity']

            # Transverse thermal conductivity (k_transverse)
            k_tr_nominal = props.get('k_transverse', k_nominal)
            dist_k_tr = self.dist_config['thermal_conductivity'].copy()
            if 'k_transverse_dist' in props:
                dist_k_tr.update(props['k_transverse_dist'])
            samples[name]['k_transverse'] = self._sample_from_dist(
                dist_k_tr, n_samples, nominal=k_tr_nominal
            )

            # Aspect ratio
            dist_AR = self.dist_config['aspect_ratio'].copy()
            if 'aspect_ratio_dist' in props:
                dist_AR.update(props['aspect_ratio_dist'])
            samples[name]['aspect_ratio'] = self._sample_from_dist(
                dist_AR, n_samples, nominal=props['aspect_ratio']
            )

            # Store nominal values for reference
            samples[name]['nominal'] = {
                'thermal_conductivity': props['thermal_conductivity'],
                'aspect_ratio': props['aspect_ratio']
            }

        return samples

    def _sample_from_dist(self, dist_config: Dict, n: int, nominal: float) -> np.ndarray:
        """
        Generate n samples from the distribution described by dist_config.
        The nominal value is used to shift the distribution.
        """
        dtype = dist_config['type']

        if dtype == 'truncnorm':
            # Parameters are specified as fractions of nominal
            scale = dist_config.get('scale', 0.1) * nominal
            loc = nominal  # Center distribution on nominal

            low = dist_config.get('low')
            high = dist_config.get('high')

            # Convert bounds to z-scores
            a = (low * nominal - loc) / scale if low is not None else -np.inf
            b = (high * nominal - loc) / scale if high is not None else np.inf

            rv = stats.truncnorm(a, b, loc=loc, scale=scale)
            samples = rv.rvs(n, random_state=self.rng)

        elif dtype == 'lognorm':
            # For lognorm, we want median = nominal
            s = dist_config.get('s', 0.3)  # shape parameter

            # For lognormal, median = exp(mu), so mu = log(nominal)
            # scale = exp(mu) = nominal
            rv = stats.lognorm(s, loc=0, scale=nominal)
            samples = rv.rvs(n, random_state=self.rng)

            # Apply bounds if given
            low = dist_config.get('low')
            high = dist_config.get('high')
            if low is not None:
                samples = np.maximum(samples, low)
            if high is not None:
                samples = np.minimum(samples, high)

        else:
            raise ValueError(f"Unknown distribution type: {dtype}")

        return samples

    def nan_model_monte_carlo(self,
                               V_f: Union[float, np.ndarray],
                               filler_name: str,
                               n_samples: int = 10000,
                               return_samples: bool = False,
                               show_progress: bool = False) -> Dict[str, Any]:
        """
        Monte Carlo simulation for Nan's model with uncertainty propagation.

        Parameters
        ----------
        V_f : float or array
            Volume fraction(s) to evaluate
        filler_name : str
            Name of the filler
        n_samples : int
            Number of Monte Carlo samples
        return_samples : bool
            If True, return all individual samples
        show_progress : bool
            If True, print progress

        Returns
        -------
        dict : Statistics for each volume fraction
        """
        if filler_name not in self.fillers:
            raise ValueError(f"Filler '{filler_name}' not found")

        props = self.fillers[filler_name]
        k_f_nominal = props['thermal_conductivity']
        ar_nominal = props['aspect_ratio']

        # Sample Kapitza radius (has its own uncertainty)
        dist_a = self.dist_config['kapitza_radius'].copy()
        a_k_samples = self._sample_from_dist(dist_a, n_samples, nominal=dist_a['loc'])

        # Sample filler properties
        filler_samples = self.sample_filler_properties(n_samples)[filler_name]
        k_f_samples = filler_samples['thermal_conductivity']
        k_f_ax_samples = filler_samples['k_axial']
        k_f_tr_samples = filler_samples['k_transverse']
        ar_samples = filler_samples['aspect_ratio']
        geom = props.get('geometry', 'fiber')
        rad = props.get('radius_nm', 5000.0)

        if isinstance(V_f, np.ndarray):
            results = {
                'mean': np.zeros_like(V_f),
                'ci_lower': np.zeros_like(V_f),
                'ci_upper': np.zeros_like(V_f),
                'std': np.zeros_like(V_f)
            }

            if return_samples:
                results['samples'] = np.zeros((len(V_f), n_samples))

            for i, vf in enumerate(V_f):
                if show_progress and i % 5 == 0:
                    print(f"    MC progress: Vf = {vf*100:.1f}%")

                k_samples = np.zeros(n_samples)
                for j in range(n_samples):
                    k_samples[j] = self.nan_model(vf, k_f_ax_samples[j],
                                                  ar_samples[j], a_k_samples[j],
                                                  k_f_transverse=k_f_tr_samples[j],
                                                  geometry=geom, radius_nm=rad)

                results['mean'][i] = np.mean(k_samples)
                results['ci_lower'][i] = np.percentile(k_samples, 2.5)
                results['ci_upper'][i] = np.percentile(k_samples, 97.5)
                results['std'][i] = np.std(k_samples)

                if return_samples:
                    results['samples'][i, :] = k_samples

        else:
            k_samples = np.zeros(n_samples)
            for j in range(n_samples):
                k_samples[j] = self.nan_model(V_f, k_f_ax_samples[j],
                                              ar_samples[j], a_k_samples[j],
                                              k_f_transverse=k_f_tr_samples[j],
                                              geometry=geom, radius_nm=rad)

            results = {
                'mean': np.mean(k_samples),
                'ci_lower': np.percentile(k_samples, 2.5),
                'ci_upper': np.percentile(k_samples, 97.5),
                'std': np.std(k_samples)
            }

            if return_samples:
                results['samples'] = k_samples

        results['n_samples'] = n_samples
        results['filler'] = filler_name
        results['random_seed'] = self.random_state

        return results

    def sequential_nan_model_monte_carlo(self,
                                           V_f_total: Union[float, np.ndarray],
                                           filler_list: List[str],
                                           fractions: Optional[List[float]] = None,
                                           n_samples: int = 10000,
                                           return_samples: bool = False,
                                           show_progress: bool = False) -> Dict[str, Any]:
        """
        Monte Carlo simulation for sequential Nan model with uncertainty propagation.

        Parameters
        ----------
        V_f_total : float or array
            Total volume fraction(s)
        filler_list : list of str
            Names of fillers in the hybrid
        fractions : list of float, optional
            Individual volume fractions (must sum to V_f_total)
        n_samples : int
            Number of Monte Carlo samples
        return_samples : bool
            If True, return all individual samples
        show_progress : bool
            If True, print progress

        Returns
        -------
        dict : Statistics for each volume fraction
        """
        if fractions is None:
            if isinstance(V_f_total, np.ndarray):
                pass  # Will handle per volume fraction
            else:
                n = len(filler_list)
                fractions = [V_f_total / n] * n

        # Sample Kapitza radius (same for all fillers in a given simulation)
        dist_a = self.dist_config['kapitza_radius'].copy()
        a_k_samples = self._sample_from_dist(dist_a, n_samples, nominal=dist_a['loc'])

        # Pre-sample all filler properties
        filler_samples_all = {}
        for filler in filler_list:
            if filler not in self.fillers:
                raise ValueError(f"Filler '{filler}' not found")
            filler_samples_all[filler] = self.sample_filler_properties(n_samples)[filler]

        if isinstance(V_f_total, np.ndarray):
            results = {
                'mean': np.zeros_like(V_f_total),
                'ci_lower': np.zeros_like(V_f_total),
                'ci_upper': np.zeros_like(V_f_total),
                'std': np.zeros_like(V_f_total)
            }

            if return_samples:
                results['samples'] = np.zeros((len(V_f_total), n_samples))

            for i, Vf in enumerate(V_f_total):
                if show_progress and i % 5 == 0:
                    print(f"    MC progress: Vf = {Vf*100:.1f}%")

                # Adjust fractions for this volume fraction
                if fractions is None:
                    current_fractions = [Vf / len(filler_list)] * len(filler_list)
                else:
                    current_fractions = fractions

                k_samples = np.zeros(n_samples)
                for j in range(n_samples):
                    k_current = self.k_m

                    for filler, vf in zip(filler_list, current_fractions):
                        if vf == 0:
                            continue

                        k_f_ax = filler_samples_all[filler]['k_axial'][j]
                        k_f_tr = filler_samples_all[filler]['k_transverse'][j]
                        ar = filler_samples_all[filler]['aspect_ratio'][j]
                        a_k = a_k_samples[j]
                        
                        f_props = self.fillers[filler]
                        geom = f_props.get('geometry', 'fiber')
                        rad = f_props.get('radius_nm', 5000.0)

                        # Nan model with current matrix
                        original_k_m = self.k_m
                        self.k_m = k_current
                        k_comp = self.nan_model(vf, k_f_ax, ar, a_k,
                                                k_f_transverse=k_f_tr,
                                                geometry=geom, radius_nm=rad)
                        self.k_m = original_k_m

                        k_current = k_comp

                    k_samples[j] = k_current

                results['mean'][i] = np.mean(k_samples)
                results['ci_lower'][i] = np.percentile(k_samples, 2.5)
                results['ci_upper'][i] = np.percentile(k_samples, 97.5)
                results['std'][i] = np.std(k_samples)

                if return_samples:
                    results['samples'][i, :] = k_samples

        else:
            if fractions is None:
                current_fractions = [V_f_total / len(filler_list)] * len(filler_list)
            else:
                current_fractions = fractions

            k_samples = np.zeros(n_samples)
            for j in range(n_samples):
                k_current = self.k_m
                for filler, vf in zip(filler_list, current_fractions):
                    if vf == 0:
                        continue
                    k_f_ax = filler_samples_all[filler]['k_axial'][j]
                    k_f_tr = filler_samples_all[filler]['k_transverse'][j]
                    ar = filler_samples_all[filler]['aspect_ratio'][j]
                    a_k = a_k_samples[j]
                    
                    f_props = self.fillers[filler]
                    geom = f_props.get('geometry', 'fiber')
                    rad = f_props.get('radius_nm', 5000.0)

                    original_k_m = self.k_m
                    self.k_m = k_current
                    k_comp = self.nan_model(vf, k_f_ax, ar, a_k,
                                            k_f_transverse=k_f_tr,
                                            geometry=geom, radius_nm=rad)
                    self.k_m = original_k_m

                    k_current = k_comp

                k_samples[j] = k_current

            results = {
                'mean': np.mean(k_samples),
                'ci_lower': np.percentile(k_samples, 2.5),
                'ci_upper': np.percentile(k_samples, 97.5),
                'std': np.std(k_samples)
            }

            if return_samples:
                results['samples'] = k_samples

        results['n_samples'] = n_samples
        results['filler_list'] = filler_list
        results['random_seed'] = self.random_state

        return results

    def convergence_test(self,
                         V_f: float,
                         filler_name: str,
                         sample_sizes: List[int] = [100, 500, 1000, 2500, 5000, 7500, 10000],
                         n_repeats: int = 5) -> Dict[str, Any]:
        """
        Test convergence of Monte Carlo simulation with increasing sample sizes.
        """
        results = {
            'sample_sizes': sample_sizes,
            'mean_stability': [],
            'ci_width_stability': []
        }

        for n in sample_sizes:
            means = []
            ci_widths = []

            for repeat in range(n_repeats):
                # Different seed for each repeat
                self.rng = np.random.default_rng(self.random_state + repeat if self.random_state else None)
                mc_result = self.nan_model_monte_carlo(V_f, filler_name, n_samples=n)
                means.append(mc_result['mean'])
                ci_widths.append(mc_result['ci_upper'] - mc_result['ci_lower'])

            results['mean_stability'].append({
                'n': n,
                'mean': np.mean(means),
                'std': np.std(means),
                'cv': np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
            })

            results['ci_width_stability'].append({
                'n': n,
                'mean': np.mean(ci_widths),
                'std': np.std(ci_widths),
                'cv': np.std(ci_widths) / np.mean(ci_widths) if np.mean(ci_widths) > 0 else 0
            })

        return results

    # ----------------------------------------------------------------------
    #  Sweep functions (with Monte Carlo)
    # ----------------------------------------------------------------------
    def sweep_volume_fraction_monte_carlo(self,
                                           volume_fractions: np.ndarray,
                                           n_samples: int = 10000,
                                           show_progress: bool = True) -> Dict:
        """
        Single‑filler sweeps with Monte Carlo uncertainty quantification.
        """
        results = {}

        print("\n    Running Monte Carlo for thermal single fillers...")
        for i, filler_name in enumerate(self.fillers.keys()):
            if show_progress:
                print(f"      {i+1}/{len(self.fillers)}: {filler_name}")

            mc_result = self.nan_model_monte_carlo(
                volume_fractions,
                filler_name,
                n_samples=n_samples,
                show_progress=show_progress
            )

            results[filler_name] = {
                'mean': mc_result['mean'],
                'ci_lower': mc_result['ci_lower'],
                'ci_upper': mc_result['ci_upper'],
                'std': mc_result['std'],
                'n_samples': n_samples
            }

            # Also include all models for completeness (using mean values)
            k_f = self.fillers[filler_name]['thermal_conductivity']
            ar = self.fillers[filler_name]['aspect_ratio']
            results[f'{filler_name}_all_models'] = {
                'Maxwell': self.maxwell_model(volume_fractions, k_f),
                'Nan': mc_result['mean'],  # Use mean from Monte Carlo
                'Series': self.series_model(volume_fractions, k_f),
                'Parallel': self.parallel_model(volume_fractions, k_f)
            }

        return results

    def sweep_hybrid_combinations_monte_carlo(self,
                                               volume_fractions: np.ndarray,
                                               filler_combinations: List[Tuple[str, ...]],
                                               distribution: str = 'equal',
                                               n_samples: int = 10000,
                                               show_progress: bool = True) -> Dict[str, Dict]:
        """
        Hybrid sweeps with Monte Carlo uncertainty quantification.
        """
        results = {}

        print("\n    Running Monte Carlo for thermal hybrid combinations...")
        for i, combo in enumerate(filler_combinations):
            key = '+'.join(combo)
            if show_progress:
                print(f"      {i+1}/{len(filler_combinations)}: {key}")

            mc_result = self.sequential_nan_model_monte_carlo(
                volume_fractions,
                combo,
                n_samples=n_samples,
                show_progress=show_progress
            )

            results[key] = {
                'mean': mc_result['mean'],
                'ci_lower': mc_result['ci_lower'],
                'ci_upper': mc_result['ci_upper'],
                'std': mc_result['std'],
                'n_samples': n_samples
            }

        return results

    # ----------------------------------------------------------------------
    #  Deterministic sweep functions (kept for backward compatibility)
    # ----------------------------------------------------------------------
    def sweep_volume_fraction_deterministic(self, volume_fractions: np.ndarray) -> Dict:
        """
        Single‑filler baseline sweeps (deterministic, nominal properties).
        """
        results = {}
        for filler_name, props in self.fillers.items():
            k_f = props.get('k_axial', props['thermal_conductivity'])
            k_f_trans = props.get('k_transverse', k_f)
            aspect_ratio = props['aspect_ratio']
            geom = props.get('geometry', 'fiber')
            rad = props.get('radius_nm', 5000.0)

            k_nan = self.nan_model(volume_fractions, k_f, aspect_ratio,
                                   k_f_transverse=k_f_trans,
                                   geometry=geom, radius_nm=rad)
            results[filler_name] = k_nan

            results[f'{filler_name}_all_models'] = {
                'Maxwell': self.maxwell_model(volume_fractions, k_f),
                'Nan': k_nan,
                'Series': self.series_model(volume_fractions, k_f),
                'Parallel': self.parallel_model(volume_fractions, k_f)
            }
        return results

    def sweep_hybrid_combinations_deterministic(self,
                                                 volume_fractions: np.ndarray,
                                                 filler_combinations: List[Tuple[str, ...]],
                                                 distribution: str = 'equal',
                                                 a_k: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Hybrid sweeps using sequential homogenisation (deterministic, nominal properties).
        """
        results = {}
        for combo in filler_combinations:
            key = '+'.join(combo)
            n = len(combo)
            k_vals = []

            for Vf in volume_fractions:
                if Vf == 0:
                    k_vals.append(self.k_m)
                else:
                    fracs = [Vf / n] * n if distribution == 'equal' else [Vf / n] * n
                    k = self.sequential_nan_model(Vf, combo, fractions=fracs, a_k=a_k)
                    k_vals.append(k)

            results[key] = np.array(k_vals)
        return results