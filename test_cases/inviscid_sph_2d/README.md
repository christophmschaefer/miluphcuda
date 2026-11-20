# INVISCID_SPH in 2D: Gresho vortex and planar Sod shock tube

This is the 2D test of the time-dependent artificial viscosity of Cullen & Dehnen (2010), switched on by
`INVISCID_SPH` (see `src/viscosity.cu`). The 1D test is `test_cases/inviscid_sph`.

The 1D test cannot reach two parts of the scheme. This test covers them:
- the limiter xi, which keeps the viscosity off in shear flows and expansions;
- the full velocity gradient tensor, with the correction matrix of `TENSORIAL_CORRECTION`.

Settings in `parameter.h`:
- `DIM 2`, `HYDRO`, `INTEGRATE_ENERGY`;
- density by kernel summation;
- `TENSORIAL_CORRECTION`;
- `INVISCID_SPH 1` with `INVISCID_SPH_START_ALPHA_MIN 1`. Alpha starts at `INVISCID_SPH_ALPHA_MIN` = 0, so every
  nonzero `av_alpha` comes from the shock indicator.

The runs use the Wendland C2 kernel with support 4 dx and the rk2_adaptive integrator.

Each set-up also runs with `INVISCID_SPH 0`, the constant alpha = 1, beta = 2, for comparison.

## Frozen outer layers

Free edges of a gas expanding into vacuum are fragile in SPH: the internal energy of the outermost particles can
become negative. The outer layers of both set-ups are therefore frozen particles (materials 2 and 3).
`boundary.cu` of this directory keeps their position, velocity and internal energy fixed. It is the
`src/boundary.cu` of the repository plus this block in `BoundaryConditionsAfterRHS`, and `build.sh` copies it into
the build.

The frozen particles take part in the kernel sums. Their density and pressure therefore follow their neighbours,
and they confine the gas.

## Set-ups

**Gresho vortex** (Gresho & Chan 1990), `gresho.0000`:
- gamma = 5/3, rho = 1.
- A stationary vortex: v_phi = 5r for r < 0.2, 2 - 5r for 0.2 < r < 0.4, 0 outside, with the balancing pressure.
- Square lattice, dx = 0.02 on [-1, 1]^2 (10000 particles), t = 1.
- Frozen where max(|x|, |y|) > 0.76.

This is a shear flow without shocks: the viscosity must stay off.

**Planar Sod shock tube**, `sod2d.0000`:
- gamma = 1.4.
- rho = 1, p = 1 for x < 0.5; rho = 0.125, p = 0.1 for x > 0.5.
- Square lattice, dx = 0.01 on [-1, 2.5] x [-1.1, 1.1] (77000 particles), masses rho dx^2, t = 0.6.
- Frozen where x < -0.85, x > 2.35 or |y| > 0.95.
- Evaluated in the strip |y| < 0.15 against the exact Riemann solution. Until t = 0.6, the waves do not reach the
  frozen ends, and disturbances from the frozen sides do not reach the strip.

## Running

```
CUDA_DIR=/usr/local/cuda-12.8 GPU_ARCH=-arch=sm_61 ./build.sh     # or: ./build.sh /path/to/miluphcuda
./run.sh
```

`run.sh` writes `run_{inviscid,standard}_{gresho,sod2d}/` and then runs `check.py`. `check.py` prints PASS or FAIL
for each check, plots `gresho.png` and `sod2d.png`, and exits with status 1 if a check fails. It analyses the gas
only, without the frozen particles.

## Checks

All alpha thresholds are for alpha_max = 1.

Gresho vortex:
- **G1**: alpha for r < 0.4 stays low: mass-weighted mean < 0.1 and maximum < 0.3.
- **G2**: the L1 error of the v_phi profile over r < 0.5 is at most 0.5 times that with the constant viscosity.
- **G3**: the peak of the v_phi profile is at least 0.7. The exact value is 1.

Sod in 2D:
- **S1**: the shock position is within 2 h of the exact one.
- **S2**: the post-shock density is within 3 % of the exact value.
- **S3**: the mean alpha in the undisturbed gas is < 0.01.
- **S4**: alpha at the shock is at least 0.15.
- **S5**: the maximum alpha in the rarefaction is < 0.05. This tests the limiter: an expansion has R ~ 1. In 1D,
  where xi = 1, alpha rises there to about 0.1.
- **S6**: the L1 density error is at most 1.25 times that with the constant viscosity.
- **S7**: alpha behind the shock decays. Its mean between contact and shock is at most 0.7 times the value of S4.

## Calibration

The thresholds come from a numpy model of the same equations, with the same frozen layers but without the
correction matrix: Wendland C2, density summation, energy equation, Monaghan viscosity with the switch, Heun steps.
Expect small differences on the GPU.

| Check | Model value |
|---|---|
| G1 alpha in the vortex | mean 0.035, maximum 0.091 |
| G2 L1 of the v_phi profile | 0.045, against 0.293 with the constant viscosity |
| G3 peak of v_phi | 0.80 |
| S3 alpha in the undisturbed gas | 0 |
| S4 alpha at the shock | 0.29 |
| S5 alpha in the rarefaction | 0 |
| S6 L1 density error | 0.0147, against 0.0171 with the constant viscosity |
| S7 alpha behind the shock | 0.105 |

Switching the limiter off in the model (xi = 1) makes S5 fail: alpha in the rarefaction rises to 0.14. G1 still
passes (mean 0.044, maximum 0.096). In the vortex, d(div v)/dt = div a - V_ab V_ba is nearly zero, so the indicator
stays low even without the limiter. G1 therefore mainly checks that velocity gradient and div a are consistent.

With the default start value (alpha = alpha_max) and free edges, the vortex kept much less of its profile in the
model: L1 0.14 instead of 0.04. That is because the full viscosity of the first ~0.3 time units has already damped
it.
