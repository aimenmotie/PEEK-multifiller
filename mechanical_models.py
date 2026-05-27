"""
mechanical_models.py
Mechanical property prediction models for PEEK composites.

Implements micromechanical models with support for uncertainty quantification
via bounded input distributions and Monte Carlo propagation. For multi‑filler
systems a true sequential homogenisation (Halpin‑Tsai applied successively)
is used, where after each filler addition the composite becomes the matrix
for the next filler.

Random orientation averaging for Mori‑Tanaka follows the scheme of [Benveniste, 1987].
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Union, Optional, Tuple, Any


class MechanicalSimulator:
    """
    Simulates mechanical properties (Young's modulus) of PEEK composites.

    Parameters
    ----------
    matrix_properties : dict
        Must contain keys: 'young_modulus' (GPa), 'poisson_ratio'.
    filler_properties : dict
        Nested dictionary. Each filler entry must contain:
            'young_modulus' (GPa, nominal value),
            'aspect_ratio' (nominal),
            optionally distribution parameters for uncertainty.
    random_state : int or np.random.Generator, optional
        Seed for reproducible sampling.
    """

    def __init__(self,
                 matrix_properties: Dict,
                 filler_properties: Dict,
                 random_state: Optional[int] = None):
        self.matrix = matrix_properties
        self.fillers = filler_properties
        self.E_m = matrix_properties['young_modulus']
        self.nu_m = matrix_properties['poisson_ratio']

        # Random generator for reproducible sampling
        self.rng = np.random.default_rng(random_state)
        self.random_state = random_state

        # Default distribution settings (can be overridden per filler)
        self.dist_config = {
            'young_modulus': {
                'type': 'truncnorm',
                'loc': 1.0,      # Will be scaled by nominal value
                'scale': 0.1,     # 10% CV
                'low': 0.5,       # Lower bound as fraction of nominal
                'high': 1.5        # Upper bound as fraction of nominal
            },
            'aspect_ratio': {
                'type': 'lognorm',
                's': 0.3,          # Shape parameter
                'low': 1.0,         # Minimum aspect ratio
                'high': None        # No upper bound by default
            }
        }
        # 'loc' and 'scale' here are for the underlying normal; final value = nominal * (loc + scale * Z)
        # For lognorm, the parameters follow scipy.stats.lognorm conventions.

    # ----------------------------------------------------------------------
    #  Single‑filler models (deterministic)
    # ----------------------------------------------------------------------

    def rule_of_mixtures(self, V_f: Union[float, np.ndarray], E_f: float,
                          efficiency: float = 0.2) -> Union[float, np.ndarray]:
        """
        Rule of mixtures (Voigt bound) with an empirical efficiency factor.
        """
        return efficiency * E_f * V_f + self.E_m * (1 - V_f)

    def halpin_tsai(self, V_f: Union[float, np.ndarray], E_f: float,
                    aspect_ratio: float, orientation: str = 'random',
                    geometry: str = 'fiber') -> Union[float, np.ndarray]:
        """
        Halpin‑Tsai equation for discontinuous reinforcements.
        [TuckerLiang1999] DOI: 10.1016/S0266-3538(98)00120-1
        [Toth2025] DOI: 10.1007/s00170-025-15738-x

        Geometry factor ξ:
          fiber:   ξ = 2·α
          platelet: ξ = (2/3)·α
        """
        # --- Geometry factor ---
        # [TuckerLiang1999]: ξ = 2α for fibers, ξ = (2/3)α for platelets
        if geometry == 'platelet':
            xi = (2.0 / 3.0) * aspect_ratio
        else:  # 'fiber' (default)
            xi = 2.0 * aspect_ratio

        numerator = (E_f / self.E_m) - 1
        denominator = (E_f / self.E_m) + xi
        eta = numerator / denominator

        E_c = self.E_m * (1 + xi * eta * V_f) / (1 - eta * V_f)

        if orientation == 'random':
            # Apply orientation factor only to the reinforcement contribution.
            # The matrix is isotropic; orientation of fillers must not reduce
            # the matrix modulus.  At V_f=0 this returns E_m exactly.
            E_c = self.E_m + 0.5 * (E_c - self.E_m)

        return E_c

    def mori_tanaka(self, V_f: Union[float, np.ndarray], E_f: float,
                    aspect_ratio: Optional[float] = None,
                    geometry: str = 'fiber') -> Union[float, np.ndarray]:
        """
        Mori‑Tanaka mean‑field homogenisation.

        .. deprecated::
            Full Eshelby orientation averaging is not implemented.
            This function returns a Halpin-Tsai aligned approximation
            and should not be used in publications.
            [Dai2004] DOI: 10.1002/pc.10125
            [TuckerLiang1999] DOI: 10.1016/S0266-3538(98)00120-1
        """
        import warnings
        warnings.warn(
            "mori_tanaka(): full Eshelby orientation averaging is not "
            "implemented; this function returns a Halpin-Tsai aligned "
            "approximation and should not be used in publications.",
            DeprecationWarning, stacklevel=2
        )
        if aspect_ratio is None or aspect_ratio < 1.1:
            # Spherical inclusions – exact isotropic result
            r = E_f / self.E_m
            E_c = self.E_m * (1 + V_f * (r - 1) /
                              (1 + (1 - V_f) * (r - 1) * (1 - 2*self.nu_m) / (1 + self.nu_m)))
        else:
            # Aligned inclusions – approximate (full orientation averaging not yet implemented)
            E_c = self.halpin_tsai(V_f, E_f, aspect_ratio, orientation='aligned', geometry=geometry)
        return E_c

    # ----------------------------------------------------------------------
    #  Multi‑filler (sequential) homogenisation (deterministic)
    # ----------------------------------------------------------------------
    def sequential_halpin_tsai(self,
                                V_f_total: Union[float, np.ndarray],
                                filler_list: List[str],
                                fractions: Optional[List[float]] = None,
                                orientation: str = 'random') -> Union[float, np.ndarray]:
        """
        Sequential Halpin‑Tsai for hybrid composites (deterministic, nominal values).

        The algorithm proceeds as:
            1. Start with matrix properties (E_m, nu_m).
            2. For each filler i (ordered arbitrarily):
                a. Compute composite properties using Halpin‑Tsai with the current matrix.
                b. Update matrix = composite (i.e., the new effective medium).
            3. The final composite properties are obtained after processing all fillers.

        Parameters
        ----------
        V_f_total : total volume fraction of all fillers (0‑1)
        filler_list : names of fillers in the hybrid
        fractions : volume fraction of each filler (must sum to V_f_total).
                    If None, equal division is assumed.
        orientation : 'random' or 'aligned'

        Returns
        -------
        Composite Young's modulus (GPa) – scalar if V_f_total is float, else array.
        """
        if fractions is None:
            n = len(filler_list)
            fractions = [V_f_total / n] * n
        else:
            if abs(sum(fractions) - V_f_total) > 1e-12:
                raise ValueError("Sum of fractions must equal V_f_total")

        if isinstance(V_f_total, np.ndarray):
            # Vectorised call – loop over each volume fraction
            return np.array([self.sequential_halpin_tsai(vf, filler_list, fractions, orientation)
                             for vf in V_f_total])

        # Scalar case
        if V_f_total == 0:
            return self.E_m

        # Start with matrix
        E_current = self.E_m
        nu_current = self.nu_m   # Poisson's ratio evolution is ignored in simple Halpin‑Tsai

        for filler_name, vf in zip(filler_list, fractions):
            if vf == 0:
                continue
            props = self.fillers[filler_name]
            E_f = props['young_modulus']
            ar = props['aspect_ratio']
            geom = props.get('geometry', 'fiber')

            # Halpin‑Tsai using current matrix
            # We need a local copy of the matrix modulus
            E_matrix = E_current
            # Temporarily replace self.E_m to reuse halpin_tsai
            original_E_m = self.E_m
            self.E_m = E_matrix
            E_comp = self.halpin_tsai(vf, E_f, ar, orientation, geometry=geom)
            self.E_m = original_E_m   # restore

            # Update matrix for next filler
            E_current = E_comp
            # (Poisson's ratio could be updated via a mixture rule, but we neglect for now)

        return E_current

    # ----------------------------------------------------------------------
    #  Uncertainty propagation helpers (Monte Carlo)
    # ----------------------------------------------------------------------
    def sample_filler_properties(self, n_samples: int) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Draw random samples of filler properties from specified distributions.

        Returns a nested dictionary: filler_name -> property_name -> array of size n_samples.
        The nominal values are used as the basis for distributions.
        """
        samples = {}
        for name, props in self.fillers.items():
            samples[name] = {}
            
            # Young's modulus
            dist_E = self.dist_config['young_modulus'].copy()
            # Override with per‑filler settings if present
            if 'young_modulus_dist' in props:
                dist_E.update(props['young_modulus_dist'])
            samples[name]['young_modulus'] = self._sample_from_dist(
                dist_E, n_samples, nominal=props['young_modulus']
            )
            
            # Aspect ratio
            dist_AR = self.dist_config['aspect_ratio'].copy()
            if 'aspect_ratio_dist' in props:
                dist_AR.update(props['aspect_ratio_dist'])
            samples[name]['aspect_ratio'] = self._sample_from_dist(
                dist_AR, n_samples, nominal=props['aspect_ratio']
            )
            
            # Also store nominal values for reference
            samples[name]['nominal'] = {
                'young_modulus': props['young_modulus'],
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

    def halpin_tsai_monte_carlo(self,
                                 V_f: Union[float, np.ndarray],
                                 filler_name: str,
                                 n_samples: int = 10000,
                                 orientation: str = 'random',
                                 return_samples: bool = False,
                                 show_progress: bool = False) -> Dict[str, Any]:
        """
        Monte Carlo simulation for Halpin-Tsai model with uncertainty propagation.

        Parameters
        ----------
        V_f : float or array
            Volume fraction(s) to evaluate
        filler_name : str
            Name of the filler (must exist in self.fillers)
        n_samples : int
            Number of Monte Carlo samples
        orientation : str
            'random' or 'aligned'
        return_samples : bool
            If True, return all individual samples (useful for debugging)
        show_progress : bool
            If True, print progress (useful for large sweeps)

        Returns
        -------
        dict : Contains for each volume fraction:
            - 'mean': mean predicted modulus
            - 'ci_lower': 2.5th percentile
            - 'ci_upper': 97.5th percentile
            - 'std': standard deviation
            - 'samples': (optional) all samples if return_samples=True
        """
        # Check if filler exists
        if filler_name not in self.fillers:
            raise ValueError(f"Filler '{filler_name}' not found in filler_properties")

        # Get filler properties
        props = self.fillers[filler_name]
        E_f_nominal = props['young_modulus']
        ar_nominal = props['aspect_ratio']

        # Pre-sample all random inputs for reproducibility
        filler_samples = self.sample_filler_properties(n_samples)[filler_name]
        E_f_samples = filler_samples['young_modulus']
        ar_samples = filler_samples['aspect_ratio']

        # Handle array input for volume fractions
        if isinstance(V_f, np.ndarray):
            results = {
                'mean': np.zeros_like(V_f),
                'ci_lower': np.zeros_like(V_f),
                'ci_upper': np.zeros_like(V_f),
                'std': np.zeros_like(V_f)
            }
            
            if return_samples:
                results['samples'] = np.zeros((len(V_f), n_samples))

            geom = props.get('geometry', 'fiber')

            for i, vf in enumerate(V_f):
                if show_progress and i % 5 == 0:
                    print(f"    MC progress: Vf = {vf*100:.1f}%")

                # Calculate modulus for each sample
                E_samples = np.zeros(n_samples)
                for j in range(n_samples):
                    E_samples[j] = self.halpin_tsai(vf, E_f_samples[j], ar_samples[j], orientation, geometry=geom)

                # Calculate statistics
                results['mean'][i] = np.mean(E_samples)
                results['ci_lower'][i] = np.percentile(E_samples, 2.5)
                results['ci_upper'][i] = np.percentile(E_samples, 97.5)
                results['std'][i] = np.std(E_samples)
                
                if return_samples:
                    results['samples'][i, :] = E_samples

        else:
            # Single volume fraction
            geom = props.get('geometry', 'fiber')
            E_samples = np.zeros(n_samples)
            for j in range(n_samples):
                E_samples[j] = self.halpin_tsai(V_f, E_f_samples[j], ar_samples[j], orientation, geometry=geom)

            results = {
                'mean': np.mean(E_samples),
                'ci_lower': np.percentile(E_samples, 2.5),
                'ci_upper': np.percentile(E_samples, 97.5),
                'std': np.std(E_samples)
            }
            
            if return_samples:
                results['samples'] = E_samples

        # Add metadata
        results['n_samples'] = n_samples
        results['filler'] = filler_name
        results['orientation'] = orientation
        results['random_seed'] = self.random_state

        return results

    def sequential_halpin_tsai_monte_carlo(self,
                                            V_f_total: Union[float, np.ndarray],
                                            filler_list: List[str],
                                            fractions: Optional[List[float]] = None,
                                            orientation: str = 'random',
                                            n_samples: int = 10000,
                                            return_samples: bool = False,
                                            show_progress: bool = False) -> Dict[str, Any]:
        """
        Monte Carlo simulation for sequential Halpin-Tsai with uncertainty propagation.

        Parameters
        ----------
        V_f_total : float or array
            Total volume fraction(s)
        filler_list : list of str
            Names of fillers in the hybrid
        fractions : list of float, optional
            Individual volume fractions (must sum to V_f_total)
        orientation : str
            'random' or 'aligned'
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
                # Will handle later per volume fraction
                pass
            else:
                n = len(filler_list)
                fractions = [V_f_total / n] * n
        else:
            if isinstance(V_f_total, np.ndarray):
                # Check first element for sum consistency
                if abs(sum(fractions) - V_f_total[0]) > 1e-12:
                    raise ValueError("Sum of fractions must equal V_f_total")
            else:
                if abs(sum(fractions) - V_f_total) > 1e-12:
                    raise ValueError("Sum of fractions must equal V_f_total")

        # Pre-sample all filler properties
        filler_samples_all = {}
        for filler in filler_list:
            if filler not in self.fillers:
                raise ValueError(f"Filler '{filler}' not found")
            filler_samples_all[filler] = self.sample_filler_properties(n_samples)[filler]

        # Handle array input
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
                    current_fractions = fractions  # Assume fractions scale with Vf_total

                # Calculate for each sample
                E_samples = np.zeros(n_samples)
                for j in range(n_samples):
                    # Start with matrix
                    E_current = self.E_m
                    
                    for filler, vf in zip(filler_list, current_fractions):
                        if vf == 0:
                            continue
                            
                        E_f = filler_samples_all[filler]['young_modulus'][j]
                        ar = filler_samples_all[filler]['aspect_ratio'][j]
                        geom = self.fillers[filler].get('geometry', 'fiber')
                        
                        # Halpin-Tsai with current matrix
                        original_E_m = self.E_m
                        self.E_m = E_current
                        E_comp = self.halpin_tsai(vf, E_f, ar, orientation, geometry=geom)
                        self.E_m = original_E_m
                        
                        E_current = E_comp
                    
                    E_samples[j] = E_current

                # Statistics
                results['mean'][i] = np.mean(E_samples)
                results['ci_lower'][i] = np.percentile(E_samples, 2.5)
                results['ci_upper'][i] = np.percentile(E_samples, 97.5)
                results['std'][i] = np.std(E_samples)
                
                if return_samples:
                    results['samples'][i, :] = E_samples

        else:
            # Single volume fraction
            if fractions is None:
                current_fractions = [V_f_total / len(filler_list)] * len(filler_list)
            else:
                current_fractions = fractions

            E_samples = np.zeros(n_samples)
            for j in range(n_samples):
                E_current = self.E_m
                for filler, vf in zip(filler_list, current_fractions):
                    if vf == 0:
                        continue
                    E_f = filler_samples_all[filler]['young_modulus'][j]
                    ar = filler_samples_all[filler]['aspect_ratio'][j]
                    geom = self.fillers[filler].get('geometry', 'fiber')
                    
                    original_E_m = self.E_m
                    self.E_m = E_current
                    E_comp = self.halpin_tsai(vf, E_f, ar, orientation, geometry=geom)
                    self.E_m = original_E_m
                    
                    E_current = E_comp
                
                E_samples[j] = E_current

            results = {
                'mean': np.mean(E_samples),
                'ci_lower': np.percentile(E_samples, 2.5),
                'ci_upper': np.percentile(E_samples, 97.5),
                'std': np.std(E_samples)
            }
            
            if return_samples:
                results['samples'] = E_samples

        # Add metadata
        results['n_samples'] = n_samples
        results['filler_list'] = filler_list
        results['orientation'] = orientation
        results['random_seed'] = self.random_state

        return results

    def convergence_test(self,
                         V_f: float,
                         filler_name: str,
                         sample_sizes: List[int] = [100, 500, 1000, 2500, 5000, 7500, 10000],
                         n_repeats: int = 5,
                         orientation: str = 'random') -> Dict[str, Any]:
        """
        Test convergence of Monte Carlo simulation by running with increasing sample sizes.

        Parameters
        ----------
        V_f : float
            Volume fraction to test
        filler_name : str
            Name of filler
        sample_sizes : list of int
            Sample sizes to test
        n_repeats : int
            Number of repeats at each sample size to estimate variability
        orientation : str
            'random' or 'aligned'

        Returns
        -------
        dict : Convergence metrics
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
                # Run Monte Carlo with different random seed each time
                self.rng = np.random.default_rng(self.random_state + repeat if self.random_state else None)
                mc_result = self.halpin_tsai_monte_carlo(V_f, filler_name, n_samples=n, orientation=orientation)
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

        Returns a dictionary with keys:
            filler_name -> {
                'young_modulus': {
                    'Halpin-Tsai': {
                        'mean': array,
                        'ci_lower': array,
                        'ci_upper': array
                    }
                },
                'enhancement_factor': array (mean),
                'n_samples': int
            }
        """
        results = {}
        
        print("\n    Running Monte Carlo for single fillers...")
        for i, filler_name in enumerate(self.fillers.keys()):
            if show_progress:
                print(f"      {i+1}/{len(self.fillers)}: {filler_name}")
            
            mc_result = self.halpin_tsai_monte_carlo(
                volume_fractions, 
                filler_name, 
                n_samples=n_samples,
                show_progress=show_progress
            )
            
            # Calculate enhancement factor from mean values
            enhancement = mc_result['mean'] / self.E_m
            
            results[filler_name] = {
                'young_modulus': {
                    'Halpin-Tsai': {
                        'mean': mc_result['mean'],
                        'ci_lower': mc_result['ci_lower'],
                        'ci_upper': mc_result['ci_upper'],
                        'std': mc_result['std']
                    }
                },
                'enhancement_factor': enhancement,
                'n_samples': n_samples
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

        Returns a dict: combination_name -> {
            'mean': array,
            'ci_lower': array,
            'ci_upper': array,
            'std': array
        }
        """
        results = {}
        
        print("\n    Running Monte Carlo for hybrid combinations...")
        for i, combo in enumerate(filler_combinations):
            key = '+'.join(combo)
            if show_progress:
                print(f"      {i+1}/{len(filler_combinations)}: {key}")
            
            mc_result = self.sequential_halpin_tsai_monte_carlo(
                volume_fractions,
                combo,
                fractions=None,  # Will be computed per Vf
                orientation='random',
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
        Single‑filler baseline sweeps (deterministic, using nominal properties).
        Returns a dictionary with keys:
            filler_name -> {
                'young_modulus': {'Halpin-Tsai': array, 'ROM': array, 'Mori-Tanaka': array},
                'tensile_strength': array,
                'enhancement_factor': array
            }
        """
        results = {}
        for filler_name, props in self.fillers.items():
            E_f = props['young_modulus']
            ar = props['aspect_ratio']
            sigma_f = props['tensile_strength']
            geom = props.get('geometry', 'fiber')

            E_ht = self.halpin_tsai(volume_fractions, E_f, ar, geometry=geom)

            results[filler_name] = {
                'young_modulus': {
                    'ROM': self.rule_of_mixtures(volume_fractions, E_f),
                    'Halpin-Tsai': E_ht,
                    'Mori-Tanaka': self.mori_tanaka(volume_fractions, E_f, aspect_ratio=ar, geometry=geom)
                },
                'tensile_strength': self.tensile_strength(volume_fractions, sigma_f, aspect_ratio=ar),
                'enhancement_factor': E_ht / self.E_m
            }
        return results

    def sweep_hybrid_combinations_deterministic(self,
                                                 volume_fractions: np.ndarray,
                                                 filler_combinations: List[Tuple[str, ...]],
                                                 distribution: str = 'equal') -> Dict[str, np.ndarray]:
        """
        Hybrid sweeps using sequential homogenisation (deterministic, nominal properties).
        Returns a dict: combination_name -> array of modulus vs volume_fractions.
        """
        results = {}
        for combo in filler_combinations:
            key = '+'.join(combo)
            n = len(combo)
            E_vals = []
            for Vf in volume_fractions:
                if Vf == 0:
                    E_vals.append(self.E_m)
                else:
                    fracs = [Vf / n] * n   # equal distribution
                    E = self.sequential_halpin_tsai(Vf, combo, fractions=fracs)
                    E_vals.append(E)
            results[key] = np.array(E_vals)
        return results

    # ----------------------------------------------------------------------
    #  Strength model (kept simple)
    # ----------------------------------------------------------------------
    def tensile_strength(self, V_f: Union[float, np.ndarray], sigma_f: float,
                          sigma_m: Optional[float] = None,
                          aspect_ratio: Optional[float] = None,
                          Lc: Optional[float] = None) -> Union[float, np.ndarray]:
        """Kelly‑Tyson based strength prediction."""
        if sigma_m is None:
            sigma_m = self.matrix['tensile_strength']
        if Lc is None or aspect_ratio is None:
            return sigma_f * V_f + sigma_m * (1 - V_f)
        else:
            eta = 1 - (Lc / (2 * aspect_ratio))
            return eta * sigma_f * V_f + sigma_m * (1 - V_f)