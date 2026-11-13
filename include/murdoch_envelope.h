/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2026 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _MURDOCH_ENVELOPE_H
#define _MURDOCH_ENVELOPE_H

#include <math.h>

/*
 * Nonlinear (hyperbolic) Mohr-Coulomb failure envelope of Maksimovic (1989, 1991) with true cohesion, as used by
 * Murdoch et al. (2026, PSJ 7, 240, eq. 3), for MURDOCH_ENVELOPE (with COLLINS_PLASTICITY_SIMPLE).
 *
 * Shear strength on the failure plane as a function of the normal stress s (compression positive):
 *
 *     tau(s) = c + s tan(phi_inf) + s (tan(phi_0) - tan(phi_inf)) / (1 + s/sigma_h)        for s >= 0
 *     tau(s) = c + s tan(phi_0)                                                            for s <  0
 *
 * c: true cohesion, phi_inf: high-stress friction angle, phi_0 >= phi_inf: low-stress friction angle, sigma_h > 0:
 * stress scale of the transition. For s < 0 the envelope is continued linearly with its slope tan(phi_0) at s = 0
 * (continuous and differentiable), down to its zero at s = -c/tan(phi_0) (tension cut-off). Without nonlinearity
 * (phi_0 = phi_inf), it is the linear Mohr-Coulomb envelope c + s tan(phi_inf).
 *
 * The yield strength is the largest Mohr circle that does not cross the envelope. With the mean pressure p (centre
 * of the circle) and the radius R = sqrt(J2) of the circle, the circle touches the (concave) envelope at the point
 * s* where the normal to the envelope passes through the centre:
 *
 *     g(s*) = s* + tau(s*) tau'(s*) = p,        R(p) = tau(s*) sqrt(1 + tau'(s*)^2).
 *
 * For a linear envelope this is the Mohr-Coulomb criterion sqrt(J2) = c cos(phi) + p sin(phi). As all pressure
 * dependent models in plasticity.cu, it is applied as a Drucker-Prager type criterion to the mean pressure p and
 * sqrt(J2) of the deviator, i.e. independent of the Lode angle. It coincides with the Mohr-Coulomb criterion for the
 * envelope at Lode angle 0 (pure shear, sigma_2 = (sigma_1 + sigma_3)/2, the plane-strain state for plastic flow
 * without dilatancy), which is the consistent choice for the deviatoric radial return of the code. For other stress
 * states it deviates from Mohr-Coulomb: for phi > asin(2 sqrt(3) - 3) = 27.6 deg it lies between the triaxial
 * extension (inner) and compression (outer) cones, e.g. for phi = 40 deg it is 32 % weaker in triaxial compression
 * and 5 % stronger in triaxial extension. Note on 2D: the deviator of the code is the 2x2 in-plane block, which is
 * not traceless (dS/dt = 2 G (edot - tr(edot)/3), with an implicit S_zz = -(S_xx + S_yy)), and J2 is computed from
 * the 2x2 block only; the criterion is therefore not exactly the in-plane Mohr circle either, unless S_xx + S_yy = 0.
 *
 * Note that friction_angle and cohesion have a different meaning than with plain COLLINS_PLASTICITY_SIMPLE, where
 * the yield strength is cohesion + mu p with mu = tan(friction_angle): for p >= 0 and c = 0 the two coincide for
 * friction_angle = asin(tan(phi_simple)), e.g. 35.26 deg for phi_simple = 30 deg; with cohesion, Y(p = 0) = c cos(phi_0)
 * here, and for p < 0 the slope (sin(phi_0) instead of 1) and the tension cut-off (-c/tan(phi_0) instead of -c)
 * differ. The tension cut-off and the linear continuation for s < 0 are an extension beyond Murdoch et al. (2026),
 * whose envelope is given for s >= 0 only; note that -c/tan(phi_0) becomes large for small phi_0.
 *
 * The functions are __host__ __device__, so that the code that runs on the GPU can be tested on the CPU.
 */

#ifndef __CUDACC__
# ifndef __host__
#  define __host__
# endif
# ifndef __device__
#  define __device__
# endif
#endif

/* envelope tau(s), tau'(s), tau''(s); dtan = tan(phi_0) - tan(phi_inf) >= 0, sigma_h > 0 if dtan > 0 */
__host__ __device__ static inline void murdoch_tau(double s, double c, double tan_inf, double dtan, double sigma_h,
                                                   double *tau, double *dtau, double *d2tau)
{
    if (s >= 0.0 && dtan > 0.0) {
        double q = 1.0 / (1.0 + s/sigma_h);
        *tau = c + s*tan_inf + s*dtan*q;
        *dtau = tan_inf + dtan*q*q;
        *d2tau = -2.0*dtan*q*q*q / sigma_h;
    } else {
        double tan_0 = tan_inf + dtan;
        *tau = c + s*tan_0;
        *dtau = tan_0;
        *d2tau = 0.0;
    }
}

/* tension cut-off: the yield strength is zero for p <= murdoch_p_zero(); the negative-pressure cap uses this value.
 * Zero of the envelope at s = -c/tan(phi_0), where also g(s) = s. Without friction (phi_0 = 0) the envelope has no
 * zero (tau = c, Tresca); then -c is used, as for COLLINS_PLASTICITY_SIMPLE without MURDOCH_ENVELOPE. */
__host__ __device__ static inline double murdoch_p_zero(double c, double tan_inf, double dtan)
{
    double tan_0 = tan_inf + dtan;
    return (tan_0 > 0.0) ? -c / tan_0 : -c;
}

/*
 * Uniqueness of the touching point: the squared distance D(s) = (s - p)^2 + tau(s)^2 from the centre (p, 0) to the
 * envelope has D'(s) = 2 (g(s) - p) and D''(s) = 2 g'(s), g'(s) = 1 + tau'^2 + tau tau''. If g' > 0 for all s, D is
 * convex, the touching point is unique and the circle does not cross the envelope anywhere else. For s < 0,
 * g' = 1 + tan(phi_0)^2 > 0. For s >= 0 the curvature term tau tau'' < 0 can win if the cohesion is large compared to
 * sigma_h or phi_0 is close to 90 deg (c = 0: only for phi_0 >~ 76-83 deg, depending on phi_inf; for the ranges of
 * Murdoch et al. 2026, table 1, the limit is c/sigma_h >~ 1.6). Host function for the check of the parameters at start-up:
 * returns min g'(s) for s >= 0 (sampled in s/sigma_h on [0, 10] in steps of 1e-4 and logarithmically up to 1e8).
 * A smaller cohesion (low-density weakening) only increases g'.
 */
static inline double murdoch_min_dg(double c, double tan_inf, double dtan, double sigma_h)
{
    double x, tau, dtau, d2tau, dg, min_dg = 1.0 + (tan_inf + dtan)*(tan_inf + dtan);
    int k;

    if (dtan <= 0.0)
        return min_dg;
    for (k = 0; k <= 100000 + 7*1000; k++) {
        x = (k <= 100000) ? 1e-4*k : 10.0*pow(10.0, (k - 100000)*1e-3);
        murdoch_tau(x*sigma_h, c, tan_inf, dtan, sigma_h, &tau, &dtau, &d2tau);
        dg = 1.0 + dtau*dtau + tau*d2tau;
        if (dg < min_dg)
            min_dg = dg;
    }
    return min_dg;
}

/*
 * Yield strength R(p) = max. sqrt(J2) for the mean pressure p (compression positive), see above.
 * c >= 0, tan_inf >= 0, dtan >= 0, sigma_h > 0 if dtan > 0, and murdoch_min_dg() > 0 (checked at start-up).
 */
__host__ __device__ static inline double murdoch_yield_strength(double p, double c, double tan_inf, double dtan,
                                                                double sigma_h)
{
    double tan_0 = tan_inf + dtan;
    double s, s_lo, s_hi, s_new, tau, dtau, d2tau, g, dg;
    int it;

    if (p <= murdoch_p_zero(c, tan_inf, dtan))
        return 0.0;

    /* linear part: the circle touches at s* <= 0, i.e. p <= g(0) = c tan_0 (or no nonlinearity at all) */
    if (dtan <= 0.0 || p <= c*tan_0)
        return (c + p*tan_0) / sqrt(1.0 + tan_0*tan_0);

    /* curved part, s* > 0. g is increasing; bracket from the tangents at s = 0 (slope tan_0) and s -> inf
     * (slope tan_inf), which bound tau from above and below for s >= 0:
     *     s (1 + tan_inf^2) + c tan_inf <= g(s) <= s (1 + tan_0^2) + c tan_0 */
    s_lo = (p - c*tan_0) / (1.0 + tan_0*tan_0);
    s_hi = (p - c*tan_inf) / (1.0 + tan_inf*tan_inf);
    /* safeguarded Newton iteration (s* > 0 here). g is evaluated with an absolute rounding error of order
     * DBL_EPSILON p, so s* is resolved only to ~ DBL_EPSILON (s + p): tolerance relative to s + p */
    s = s_lo;
    for (it = 0; it < 100; it++) {
        murdoch_tau(s, c, tan_inf, dtan, sigma_h, &tau, &dtau, &d2tau);
        g = s + tau*dtau - p;
        if (g == 0.0)
            break;
        if (g < 0.0)
            s_lo = s;
        else
            s_hi = s;
        dg = 1.0 + dtau*dtau + tau*d2tau;
        s_new = (dg > 0.0) ? s - g/dg : 0.5*(s_lo + s_hi);
        if (!(s_new >= s_lo && s_new <= s_hi))   /* Newton step outside the bracket (or NaN): bisection */
            s_new = 0.5*(s_lo + s_hi);
        if (fabs(s_new - s) <= 1e-15*(s_new + p) || s_hi - s_lo <= 1e-15*(s_hi + p)) {
            s = s_new;
            break;
        }
        s = s_new;
    }
    /* radius as the distance from (p, 0) to the touching point: equal to tau sqrt(1 + tau'^2) at s*, but stationary
     * in s there, i.e. insensitive to the remaining error of s */
    murdoch_tau(s, c, tan_inf, dtan, sigma_h, &tau, &dtau, &d2tau);
    return sqrt((s - p)*(s - p) + tau*tau);
}

#endif
