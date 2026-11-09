# Porous materials with the p-alpha model

This is a practical guide. The equations are in the PDF documentation (section on porosity), the switches in
[compile-time-options.md](compile-time-options.md#porosity), all material keys in
[CREATE-MATERIAL-CONFIG.md](../material-config/CREATE-MATERIAL-CONFIG.md#porosity-model-p-alpha).

The p-alpha model (Herrmann 1969; Jutzi et al. 2008) describes the porous material by the distention
α = ρ_s/ρ ≥ 1, the ratio of the matrix density ρ_s to the bulk density ρ. The pressure is P = P_s(ρ_s, e)/α, where
P_s is the EOS of the matrix material. Compaction follows the crush curve α_cc(P): when the pressure of a particle
reaches the crush curve, α decreases along it. Below the crush curve the material responds elastically. A fully
compacted particle has α = 1.

## 1. Compile time

In `include/parameter.h`:

    #define PALPHA_POROSITY 1
    #define STRESS_PALPHA_POROSITY 1        // for SOLID: the deviatoric stress depends on alpha as well
    #define PALPHA_F_ACTS_ON_ROTATION 0     // 0 recommended, see below
    #define PALPHA_ELASTIC_RELOADING 1      // cms_devel, 1 recommended
    #define PALPHA_ELASTIC_RELOADING_TOL 1e-6
    #define PALPHA_RETURN_MAPPING 0         // cms_devel, optional, see Sect. 4

* `PALPHA_F_ACTS_ON_ROTATION`: the factor f = d ln ρ_s / d ln ρ relates the volumetric strain rate of the matrix to that
  of the bulk material. Jutzi et al. (2008) apply f to the whole Hooke law including the Jaumann rotation terms.
  With 0, f acts only on the volumetric part, which leaves the (purely kinematic) rotation terms unscaled and
  reproduces rigid rotation of a porous body exactly. Use 1 only for comparisons with the Bern SPH code.
* `PALPHA_ELASTIC_RELOADING`: with 1, α keeps the memory of the maximum pressure, α = α_cc(P_max). A particle that has
  been unloaded lies below the crush curve and reloads elastically until its pressure reaches the crush curve again.
  With 0 (Jutzi et al. 2008, eqs. 22–23), only unloading is elastic and every pressure increase above
  `porjutzi_p_elastic` compacts, including numerical pressure noise, so that α ratchets down under oscillating
  pressure.

Several materials can be porous with different parameters; non-porous materials in the same simulation use their
normal EOS types.

## 2. Material config

A porous material uses one of the p-alpha EOS types; the parameters of the matrix EOS are given as for the non-porous
material:

| `eos.type` | Matrix EOS | Matrix reference density |
|---|---|---|
| 5 (`EOS_TYPE_JUTZI`) | Tillotson | `till_rho_0` |
| 6 (`EOS_TYPE_JUTZI_MURNAGHAN`) | Murnaghan | `rho_0` |
| 13 (`EOS_TYPE_JUTZI_ANEOS`) | ANEOS table | `aneos_rho_0` |

The reference density is always the density of the **matrix** material, not the bulk density.

Crush-curve parameters (in the `eos` group):

| Key | Meaning |
|---|---|
| `crushcurve_style` | 0: quadratic (default), 1: steep two-part curve of Jutzi et al. (2008); 2 and 4: experimental curves with hard-coded constants (Blum et al. 2023, Malamud 2023) |
| `porjutzi_p_elastic` | pressure at which compaction starts, p_e |
| `porjutzi_p_transition` | transition pressure p_t (style 1) |
| `porjutzi_p_compacted` | pressure of full compaction, p_s |
| `porjutzi_alpha_0` | distention on the crush curve at p_e, α_cc(p_e) = α₀ |
| `porjutzi_alpha_e` | α_e of the original curve (style 1); the curve is rescaled by (α₀−1)/(α_e−1) such that α_cc(p_e) = α₀ |
| `porjutzi_alpha_t` | distention at p_t (style 1, before rescaling) |
| `porjutzi_n1`, `porjutzi_n2` | exponents (style 1) |
| `cs_porous` | sound speed of the uncompacted material, default 50 % of the bulk sound speed of the matrix |

The crush curves are

* style 0: α_cc(P) = 1 + (α₀ − 1) (p_s − P)² / (p_s − p_e)² for p_e < P < p_s,
* style 1: α_cc(P) = 1 + s (α_t − 1) ((p_s − P)/(p_s − p_e))^n₂ + s (α_e − α_t) ((p_t − P)/(p_t − p_e))^n₁ for
  p_e < P < p_s, where the second term is present only for P < p_t and s = (α₀ − 1)/(α_e − 1),

with α_cc = α₀ for P ≤ p_e and α_cc = 1 for P ≥ p_s.

Example (crush curve fitted for pumice, Jutzi et al. 2008):

    eos = {
        type = 5
        ...                         # Tillotson parameters of the matrix, till_rho_0 = matrix density
        crushcurve_style = 1
        porjutzi_p_elastic = 1.0e6
        porjutzi_p_transition = 6.80e7
        porjutzi_p_compacted = 2.13e8
        porjutzi_alpha_0 = 3.0
        porjutzi_alpha_e = 4.64
        porjutzi_alpha_t = 1.90
        porjutzi_n1 = 12.0
        porjutzi_n2 = 3.0
    }

The sound speed is interpolated linearly in α between the sound speed of the matrix (α = 1) and `cs_porous`
(α = α₀). This sound speed enters the Courant condition, so a small `cs_porous` gives larger time steps.

The strength models work with porous materials. `LOW_DENSITY_WEAKENING` is implemented for EOS type 5 only and uses
the matrix density.

## 3. Initial conditions

For each particle of a porous material, the input file contains

* the distention `alpha_jutzi` (column after the stress tensor, check `./miluphcuda -Y`),
* a pressure column; it only initializes the "previous" pressure that some integrators use for their
  pressure-change check (the pressure itself is computed from the EOS); set it to 0,
* the **bulk** density ρ = ρ_s/α, where ρ_s is the matrix density at which the matrix pressure of the initial state
  is what you want (for a pressure-free state with e = 0, ρ_s = matrix reference density, so ρ = ρ₀/α),
* the mass m = ρ V of the porous particle.

The initial distention is normally α₀ (on the crush curve at p_e). A particle with α < α_cc(P) starts below the
crush curve and responds elastically until its pressure reaches the crush curve. A particle with α > α_cc(P) is
above the crush curve; with `PALPHA_RETURN_MAPPING` it is compacted onto the crush curve in the first step, and the
number of such particles is reported on stderr.

## 4. Time integration of α

There are two ways to determine α:

**Integration of dα/dt (default, `PALPHA_RETURN_MAPPING 0`).** dα/dt follows from the chain rule along the crush
curve (Jutzi et al. 2008, eq. 25) and is integrated together with all other quantities. The compaction then depends on
the time step: with too large steps the particles overshoot or lag behind the crush curve. Use

* `RK2_LIMIT_ALPHA_CHANGE 1` (limit `RK2_MAX_ALPHA_CHANGE`) and/or `RK2_LIMIT_PRESSURE_CHANGE 1` in
  `include/rk2adaptive.h` for `-I rk2_adaptive`,
* `KDK_MAX_ALPHA_CHANGE` in `include/kdk.h` for `-I kdk`,

and check the result with `utils/postprocessing/p_alpha_model/p_alpha_convergence.py`.

**Return mapping (`PALPHA_RETURN_MAPPING 1`, cms_devel).** α is not integrated. For each evaluated state, α is
determined from ρ and e: if the state with the distention α_hist of the last accepted step lies beyond the crush curve,
α is the root of α − α_cc(P_s(αρ, e)/α) in [1, α_hist] (Newton iteration with bisection safeguard); otherwise α =
α_hist (elastic). α then lies on or below the crush curve independently of the time step and the integrator, and the
corresponding changes of S and of the porous damage are applied exactly. `dalphadt` is 0 in the output, and the
α-change limits have no effect (the pressure-change check remains). Requires `PALPHA_ELASTIC_RELOADING 1`.

## 5. Damage

With `FRAGMENTATION`, porous materials additionally get a porous damage that grows with compaction (Jutzi et al.
2008, eq. 57), D_p^{1/DIM} = [(1 − (α−1)/(α₀−1) + δ)^{1/DIM} − δ^{1/DIM}] / [(1+δ)^{1/DIM} − δ^{1/DIM}] with δ = 0.01.
The total damage is D = min(1, D_t + D_p) with the tensile (Grady-Kipp) damage D_t. Output:
`/DIM_root_of_damage_porjutzi`, `/DIM_root_of_damage_tensile`, `/damage_total`.

## 6. Output

HDF5: `/alpha_jutzi`, `/dalphadt`, `/p` (bulk pressure), with `FRAGMENTATION` the damage fields above.

## 7. Fast integration scheme

With `FAST_INTEGRATION_SCHEME`, a porous material that reaches its transition time is switched to EOS type 6 with a
linear matrix EOS (Murnaghan with n = 1) and the bulk modulus and reference density from its `fast_integration`
group; `cs_porous` is set to the matrix sound speed of the new medium. The crush-curve parameters are kept.

## Other porosity models

* Sirono model, modified by Geretshauser et al. (`SIRONO_POROSITY`, EOS type 10), for highly porous aggregates of
  dust; parameters `porsirono_*`.
* ε-α model (`EPSALPHA_POROSITY`, EOS type 11, Wünnemann et al. 2006), compaction as function of the volumetric
  strain; parameters `porepsilon_*`, `utils/postprocessing/eps_alpha_model/` for diagnostics.

These are less tested than the p-alpha model.
