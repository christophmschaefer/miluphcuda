# INVISCID_SPH: Sod shock tube

Test of the time-dependent artificial viscosity of Cullen & Dehnen (2010, MNRAS 408, 669), compile-time switch
`INVISCID_SPH` (see `src/viscosity.cu`). Every particle carries its own viscosity coefficient alpha_i
(HDF5 output `av_alpha`). Ahead of a shock, alpha_i rises to the material alpha (`artificial_viscosity.alpha` in
`material.cfg`, here 1). Away from shocks it decays towards `INVISCID_SPH_ALPHA_MIN` (0), on the time scale
h/(2 `INVISCID_SPH_DECAY_L` v_sig).

The set-up is the Sod shock tube in 1D, as in `test_cases/shocktube`:
- ideal gas with gamma = 1.4;
- left state rho = 1, p = 1; right state rho = 0.125, p = 0.1;
- discontinuity at x = 0.5;
- 3375 particles of equal mass on [-1, 2];
- smoothing length 0.01, end time t = 0.2.

The same set-up is also run with `INVISCID_SPH 0`, the constant alpha = 1, beta = 2, for comparison.

## Files

| File | Purpose |
|---|---|
| `parameter.h` | `include/parameter.h` of the repository with `DIM 1`, `HYDRO`, `INTEGRATE_ENERGY`, density by kernel summation, `AVERAGE_KERNELS`, `INVISCID_SPH 1` |
| `material.cfg` | ideal gas (EoS type 9) |
| `initial_conditions.py` | writes `sod.0000` (ascii: x v m e material_type) |
| `build.sh` | builds `miluphcuda` (`INVISCID_SPH 1`) and `miluphcuda_standard` (`INVISCID_SPH 0`) from the source tree with this `parameter.h` |
| `run.sh` | runs both, 20 outputs up to t = 0.2, then `check.py` |
| `check.py` | exact Riemann solution, checks, plot `sod.png` |

## Running

```
CUDA_DIR=/usr/local/cuda-12.8 GPU_ARCH=-arch=sm_61 ./build.sh     # or: ./build.sh /path/to/miluphcuda
./run.sh
```

`run.sh` runs rk2_adaptive with `-Q 1e-6`, writing to `run_inviscid/` and `run_standard/`. It should take well under a minute per run.

## What is checked

`check.py` uses the last snapshot and prints PASS or FAIL for each check. It exits with status 1 if one fails.

1. **Shock position**: within 2 h of the exact position, 0.8504.
2. **Post-shock density**: the plateau between contact and shock is within 3 % of the exact value, 0.2656.
3. **Decay of alpha**: the undisturbed left gas has no shock indicator, so alpha decays from 1 as exp(-t/tau), with
   tau = h/(2 l c_L) = 0.0845. The mean av_alpha there must be within 10 % of exp(-0.2/0.0845) = 0.094. This
   checks the decay rate and that nothing switches the viscosity on in a gas at rest.
4. **Alpha at the shock**: max av_alpha within 3 h of the shock is at least twice the decayed value ahead of it
   (0.12). This shows the shock indicator works.
5. **Alpha in the rarefaction**: max av_alpha in the rarefaction fan is below twice the value of check 3. A smooth
   expansion must not switch the viscosity on.
6. **Density error against the standard run**: L1 error of the density over [0, 1] at most 1.25 times that of the
   standard viscosity (only with `run_standard/`).

The thresholds were set with a numpy model of the same equations: cubic spline kernel, density summation, energy
equation, Monaghan viscosity with the switch, second-order time stepping. In that model:
- L1(rho) is 0.0027, against 0.0029 with the constant viscosity;
- alpha at the shock is about 0.3;
- alpha is 0.097 in the undisturbed gas and at most 0.11 in the rarefaction fan.

Expect small differences to the GPU run.

The alpha of the shock indicator scales with h^2 A/(v_sig^2 + h^2 A). With h the support radius of the kernel, as
here, it reaches about 0.3 for this weak shock, not alpha_max. Stronger shocks give larger values.
