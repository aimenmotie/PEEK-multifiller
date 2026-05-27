"""
test_nan_unit.py
Unit tests for nan_anisotropic.py

Tests verify the mathematical identities and physical limits from:
  [Nan1997] DOI: 10.1063/1.365209, Eq. 8-10, 23
  [Chen2016] DOI: 10.1016/j.progpolymsci.2016.03.001, §3.2
"""

import sys
import unittest
import numpy as np
from nan_anisotropic import depolarization_factors, nan_thermal_conductivity


class TestNanPhysics(unittest.TestCase):
    
    def test_1_depolarization_sum_rule(self):
        """
        Test 1: 2·L_11 + L_33 = 1 for arbitrary aspect ratios and both geometries.
        [Nan1997] Eq. 8 sum rule.
        """
        p_values = [1.1, 2.0, 5.0, 10.0, 100.0, 1000.0]
        
        print("Running Test 1 (Depolarization sum rule)...")
        for p in p_values:
            for geom in ['fiber', 'platelet']:
                L_11, L_33 = depolarization_factors(p, geom)
                sum_rule = 2.0 * L_11 + L_33
                self.assertAlmostEqual(sum_rule, 1.0, places=9,
                    msg=f"Sum rule failed for aspect ratio {p} and geometry {geom}: {sum_rule}")
        print("PASS: Test 1 Passed: Depolarization sum rule 2·L_11 + L_33 = 1 holds.")

    def test_2_sphere_limit(self):
        """
        Test 2: Sphere limit (aspect ratio p ≈ 1.0) → (1/3, 1/3).
        [Nan1997] below Eq. 10.
        """
        print("Running Test 2 (Sphere limit L_ii = 1/3)...")
        
        # Test exact 1.0
        L_11, L_33 = depolarization_factors(1.0, 'fiber')
        self.assertAlmostEqual(L_11, 1.0/3.0, places=9)
        self.assertAlmostEqual(L_33, 1.0/3.0, places=9)
        
        # Test close to 1.0
        L_11_c, L_33_c = depolarization_factors(1.0000001, 'fiber')
        self.assertAlmostEqual(L_11_c, 1.0/3.0, places=6)
        self.assertAlmostEqual(L_33_c, 1.0/3.0, places=6)
        
        L_11_p, L_33_p = depolarization_factors(1.0000001, 'platelet')
        self.assertAlmostEqual(L_11_p, 1.0/3.0, places=6)
        self.assertAlmostEqual(L_33_p, 1.0/3.0, places=6)
        
        print("PASS: Test 2 Passed: Sphere depolarization factors reduce to 1/3.")

    def test_3_fiber_needle_limit(self):
        """
        Test 3: Fiber aspect ratio p -> large → L_33 -> 0, L_11 -> 1/2.
        [Nan1997] Eq. 9 prolate limit.
        """
        print("Running Test 3 (Fiber needle limit)...")
        p = 10000.0
        L_11, L_33 = depolarization_factors(p, 'fiber')
        
        self.assertLess(L_33, 1e-4, f"L_33 is too large for fiber needle: {L_33}")
        self.assertAlmostEqual(L_11, 0.5, places=4)
        print(f"PASS: Test 3 Passed: Large fiber needle depolarization factors (L_11={L_11:.6f}, L_33={L_33:.6e}).")

    def test_4_platelet_disc_limit(self):
        """
        Test 4: Platelet aspect ratio p -> large → L_33 -> 1, L_11 -> 0.
        [Nan1997] Eq. 10 oblate limit.
        """
        print("Running Test 4 (Platelet disc limit)...")
        p = 10000.0
        L_11, L_33 = depolarization_factors(p, 'platelet')
        
        self.assertGreater(L_33, 0.999, f"L_33 is too small for platelet disc: {L_33}")
        self.assertLess(L_11, 0.001, f"L_11 is too large for platelet disc: {L_11}")
        print(f"PASS: Test 4 Passed: Large platelet disc depolarization factors (L_11={L_11:.6e}, L_33={L_33:.6f}).")

    def test_5_vf_zero_boundary(self):
        """
        Test 5: V_f = 0.0 returns matrix conductivity k_m exactly.
        """
        print("Running Test 5 (V_f = 0 boundary)...")
        k_m = 0.29
        k_eff = nan_thermal_conductivity(V_f=0.0, k_m=k_m, k_f_axial=100.0,
                                         k_f_transverse=50.0, aspect_ratio=10.0,
                                         geometry='fiber', radius_nm=100.0, a_K_nm=0.1)
        self.assertEqual(k_eff, k_m)
        print("PASS: Test 5 Passed: V_f = 0 returns neat matrix conductivity exactly.")

    def test_6_isotropic_sphere_maxwell_garnett(self):
        """
        Test 6: Isotropic sphere without Kapitza resistance (a_K = 0)
        recovers the Maxwell-Garnett formula within 0.1%.
        [Chen2016] Eq. 5: Maxwell-Garnett limit.
        """
        print("Running Test 6 (Maxwell-Garnett recovery for spheres)...")
        
        V_f = 0.15
        k_m = 0.29
        k_f = 50.0
        
        # Nan 1997 calculation
        # Sphere (p=1), no Kapitza resistance (a_K=0)
        k_nan = nan_thermal_conductivity(
            V_f=V_f, k_m=k_m, k_f_axial=k_f, k_f_transverse=k_f,
            aspect_ratio=1.0, geometry='fiber', radius_nm=10.0, a_K_nm=0.0
        )
        
        # Analytical Maxwell-Garnett calculation:
        # beta = (k_f - k_m) / (k_f + 2 * k_m)
        # k_eff = k_m * (1 + 2 * V_f * beta) / (1 - V_f * beta)
        beta = (k_f - k_m) / (k_f + 2.0 * k_m)
        k_mg = k_m * (1.0 + 2.0 * V_f * beta) / (1.0 - V_f * beta)
        
        relative_error = abs(k_nan - k_mg) / k_mg
        print(f"  Nan 1997: {k_nan:.6f} W/mK | Maxwell-Garnett: {k_mg:.6f} W/mK")
        print(f"  Relative error: {relative_error:.6e}")
        
        self.assertLess(relative_error, 0.001,
                        f"Nan 1997 sphere model diverges from Maxwell-Garnett by {relative_error:.2%}")
        print("PASS: Test 6 Passed: Isotropic sphere without Kapitza recovers Maxwell-Garnett within 0.1%.")

    def test_7_monotonicity(self):
        """
        Test 7: Verification of physical monotonicity constraints.
        Adding more conductive filler (or increasing filler conductivity) must increase k_eff.
        """
        print("Running Test 7 (Monotonicity constraints)...")
        
        # 7a: Monotonicity with respect to V_f
        k_vf1 = nan_thermal_conductivity(V_f=0.05, k_m=0.29, k_f_axial=100.0,
                                         k_f_transverse=10.0, aspect_ratio=10.0,
                                         geometry='fiber', radius_nm=100.0, a_K_nm=0.1)
        k_vf2 = nan_thermal_conductivity(V_f=0.20, k_m=0.29, k_f_axial=100.0,
                                         k_f_transverse=10.0, aspect_ratio=10.0,
                                         geometry='fiber', radius_nm=100.0, a_K_nm=0.1)
        self.assertGreater(k_vf2, k_vf1, "k_eff should increase as V_f increases.")
        
        # 7b: Monotonicity with respect to filler conductivity
        k_kf1 = nan_thermal_conductivity(V_f=0.10, k_m=0.29, k_f_axial=10.0,
                                         k_f_transverse=10.0, aspect_ratio=10.0,
                                         geometry='fiber', radius_nm=100.0, a_K_nm=0.1)
        k_kf2 = nan_thermal_conductivity(V_f=0.10, k_m=0.29, k_f_axial=100.0,
                                         k_f_transverse=100.0, aspect_ratio=10.0,
                                         geometry='fiber', radius_nm=100.0, a_K_nm=0.1)
        self.assertGreater(k_kf2, k_kf1, "k_eff should increase as filler conductivity increases.")
        
        print("PASS: Test 7 Passed: Physics monotonicity holds.")

    def test_8_stability_filler_library(self):
        """
        Test 8: Numerical stability check for the full filler library parameters
        across a sweep of volume fractions V_f ∈ [0.0, 0.3].
        """
        print("Running Test 8 (Filler library stability check)...")
        
        # Nominal properties from Manuscript Table 1
        library = {
            'Carbon Fiber': {
                'k_ax': 50.0, 'k_tr': 5.0, 'ar': 20.0, 'geom': 'fiber', 'rad': 5000.0
            },
            'Graphene Nanoplatelets': {
                'k_ax': 3000.0, 'k_tr': 5.0, 'ar': 1000.0, 'geom': 'platelet', 'rad': 10.0
            },
            'Glass Fiber': {
                'k_ax': 1.0, 'k_tr': 1.0, 'ar': 15.0, 'geom': 'fiber', 'rad': 5000.0
            },
            'Carbon Nanotubes': {
                'k_ax': 2000.0, 'k_tr': 20.0, 'ar': 1000.0, 'geom': 'fiber', 'rad': 5.0
            }
        }
        
        v_fractions = np.linspace(0.0, 0.3, 31)
        k_m = 0.29
        a_K = 0.1
        
        for name, props in library.items():
            for vf in v_fractions:
                k_eff = nan_thermal_conductivity(
                    V_f=vf, k_m=k_m, k_f_axial=props['k_ax'], k_f_transverse=props['k_tr'],
                    aspect_ratio=props['ar'], geometry=props['geom'],
                    radius_nm=props['rad'], a_K_nm=a_K
                )
                
                # Check for NaN / Inf / Negatives
                self.assertFalse(np.isnan(k_eff), f"NaN encountered for {name} at V_f={vf}")
                self.assertFalse(np.isinf(k_eff), f"Inf encountered for {name} at V_f={vf}")
                self.assertGreaterEqual(k_eff, k_m, f"Composite conductivity ({k_eff}) is below physical floor k_m for {name} at V_f={vf}")
                
        print("PASS: Test 8 Passed: Full filler library is stable; no NaNs, Infs, or negative values.")


class TestHalpinTsai(unittest.TestCase):

    def setUp(self):
        from mechanical_models import MechanicalSimulator
        matrix = {'young_modulus': 3.6, 'tensile_strength': 100,
                  'thermal_conductivity': 0.25, 'poisson_ratio': 0.40}
        self.fillers = {
            'Carbon Fiber': {'young_modulus': 230, 'tensile_strength': 3500,
                'thermal_conductivity': 20.0, 'aspect_ratio': 20, 'geometry': 'fiber',
                'poisson_ratio': 0.20},
            'Glass Fiber': {'young_modulus': 72, 'tensile_strength': 2000,
                'thermal_conductivity': 1.0, 'aspect_ratio': 15, 'geometry': 'fiber',
                'poisson_ratio': 0.22},
            'Carbon Nanotubes': {'young_modulus': 1000, 'tensile_strength': 10000,
                'thermal_conductivity': 680.0, 'aspect_ratio': 1000, 'geometry': 'fiber',
                'poisson_ratio': 0.20},
        }
        self.mech = MechanicalSimulator(matrix, self.fillers)

    def test_9_vf_zero_returns_matrix_modulus(self):
        """
        Test 9: At V_f=0, random-orientation Halpin-Tsai must return E_m exactly.
        Regression test for the orientation-factor bug.
        """
        print("Running Test 9 (HT V_f=0 boundary)...")
        E_m = 3.6
        for name, props in self.fillers.items():
            E_c = self.mech.halpin_tsai(0.0, props['young_modulus'],
                                         props['aspect_ratio'], 'random',
                                         props.get('geometry', 'fiber'))
            self.assertAlmostEqual(E_c, E_m, places=10,
                msg=f"V_f=0 must return E_m for {name}, got {E_c}")
        print("PASS: Test 9 Passed: V_f=0 returns E_m exactly for all fillers.")

    def test_10_monotonicity_with_volume_fraction(self):
        """
        Test 10: E_c must increase monotonically with V_f for all fillers.
        """
        print("Running Test 10 (HT monotonicity)...")
        vf_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
        for name, props in self.fillers.items():
            prev_E = 0.0
            for vf in vf_values:
                E_c = self.mech.halpin_tsai(vf, props['young_modulus'],
                                             props['aspect_ratio'], 'random',
                                             props.get('geometry', 'fiber'))
                self.assertGreaterEqual(E_c, prev_E,
                    f"E_c must increase with V_f for {name} at V_f={vf}")
                prev_E = E_c
        print("PASS: Test 10 Passed: E_c increases monotonically with V_f.")

    def test_11_aligned_exceeds_random(self):
        """
        Test 11: For V_f > 0, aligned modulus must exceed random modulus.
        """
        print("Running Test 11 (aligned > random)...")
        for name, props in self.fillers.items():
            E_rand = self.mech.halpin_tsai(0.15, props['young_modulus'],
                                            props['aspect_ratio'], 'random',
                                            props.get('geometry', 'fiber'))
            E_alig = self.mech.halpin_tsai(0.15, props['young_modulus'],
                                            props['aspect_ratio'], 'aligned',
                                            props.get('geometry', 'fiber'))
            self.assertGreater(E_alig, E_rand,
                f"Aligned must exceed random for {name}")
        print("PASS: Test 11 Passed: Aligned orientation > random for all fillers.")


if __name__ == '__main__':
    unittest.main()
