#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

extern "C" {

    /**
     * @brief Computes the effective thermal conductivity using the General Effective Media (GEM) equation
     * (McLachlan equation) which accounts for percolation.
     *
     * Reference (DOI): https://doi.org/10.1063/1.338274
     * "An equation for the conductivity of binary mixtures with anisotropic grain structures"
     * D. S. McLachlan
     *
     * Formula:
     * (1 - V_f) * (k_m ^ (1/t) - k_eff ^ (1/t)) / (k_m ^ (1/t) + A * k_eff ^ (1/t)) +
     * V_f * (k_f ^ (1/t) - k_eff ^ (1/t)) / (k_f ^ (1/t) + A * k_eff ^ (1/t)) = 0
     *
     * Where A = (1 - phi_c) / phi_c, and phi_c is the percolation threshold.
     */
    double solve_mclachlan(double k_m, double k_f, double V_f, double phi_c, double t) {
        if (V_f == 0) return k_m;
        if (V_f >= 1) return k_f;

        double A = (1.0 - phi_c) / phi_c;

        // Use bisection method to solve the non-linear equation for k_eff
        double low = k_m;
        double high = k_f;
        double k_eff = (low + high) / 2.0;

        for (int i = 0; i < 100; ++i) { // max 100 iterations
            k_eff = (low + high) / 2.0;

            double term1 = (1.0 - V_f) * (pow(k_m, 1.0/t) - pow(k_eff, 1.0/t)) / (pow(k_m, 1.0/t) + A * pow(k_eff, 1.0/t));
            double term2 = V_f * (pow(k_f, 1.0/t) - pow(k_eff, 1.0/t)) / (pow(k_f, 1.0/t) + A * pow(k_eff, 1.0/t));

            double f = term1 + term2;

            if (std::abs(f) < 1e-7) break;

            if (f > 0) {
                low = k_eff;
            } else {
                high = k_eff;
            }
        }
        return k_eff;
    }

    /**
     * @brief Monte Carlo simulation for percolation-based thermal conductivity.
     *
     * @param v_f Array of volume fractions.
     * @param n_vf Number of volume fractions.
     * @param k_m Matrix thermal conductivity.
     * @param k_f_mean Filler thermal conductivity mean.
     * @param k_f_std Filler thermal conductivity std dev.
     * @param phi_c Percolation threshold.
     * @param t Critical exponent.
     * @param n_samples Number of Monte Carlo samples.
     * @param out_mean Array to store mean results.
     * @param out_lower Array to store lower bounds.
     * @param out_upper Array to store upper bounds.
     */
    void percolation_monte_carlo(
        const double* v_f, int n_vf,
        double k_m, double k_f_mean, double k_f_std,
        double phi_c, double t,
        int n_samples,
        double* out_mean, double* out_lower, double* out_upper
    ) {
        std::random_device rd;
        std::mt19937 gen(rd()); // using random_device for true randomness
        std::normal_distribution<double> dist_kf(k_f_mean, k_f_std);

        for (int i = 0; i < n_vf; ++i) {
            double current_vf = v_f[i];
            std::vector<double> results(n_samples);

            double sum = 0.0;
            for (int j = 0; j < n_samples; ++j) {
                // Sample filler conductivity, ensuring it's positive
                double sampled_kf = std::max(0.1, dist_kf(gen));

                double k_eff = solve_mclachlan(k_m, sampled_kf, current_vf, phi_c, t);
                results[j] = k_eff;
                sum += k_eff;
            }

            out_mean[i] = sum / n_samples;

            // Calculate percentiles
            std::sort(results.begin(), results.end());
            int idx_lower = std::max(0, static_cast<int>(0.025 * n_samples));
            int idx_upper = std::min(n_samples - 1, static_cast<int>(0.975 * n_samples));

            out_lower[i] = results[idx_lower];
            out_upper[i] = results[idx_upper];
        }
    }
}
