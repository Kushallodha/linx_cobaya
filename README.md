# linx_cobaya

Author: Kushal Lodha

Cobaya theory that replaces a Boltzmann code's static BBN table with a live
[LINX](https://arxiv.org/abs/2408.14538) network solve. Each step LINX is given
`(ombh2, dNeff)` and returns the helium mass fraction `YHe` and the final
`N_eff`, which are handed to **CAMB or CLASS** as inputs. We also provide `Y_p` and `D/H`
for a BBN abundance likelihood.

## Install

LINX is not pip-installable. Clone it separately and point `linx_path` at the
checkout (the directory that contains the `linx` package).

Python dependencies of the wrapper:

- `cobaya`
- `camb` (imported at startup for `camb.bbn.ypBBN_to_yhe`, including when the
  Boltzmann solver in the run is CLASS)
- `classy`, only if that run uses CLASS
- `jax`, `numpy`, `scipy`
- `diffrax`, `equinox`, `interpax`

`python_path` must be the directory that contains this `linx_cobaya` package
(the root of this repo).

```yaml
python_path: /path/to/linx_cobaya

theory:
  linx_cobaya.LinxBBN:
    linx_path: /path/to/LINX
likelihood:
  linx_cobaya.BBNAbundances:
```

Importing LINX turns on JAX x64 for the whole process (`linx/const.py`).

For MPI runs, keep JAX on CPU and stop each rank from spawning its own thread
pool:

```bash
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false"
```

## CAMB

[`examples/camb.yaml`](examples/camb.yaml) is Planck 2018 TTTEEE + low-l TT/EE
plus the BBN abundance likelihood. `camb` has `requires: [YHe, nnu]`.

## CLASS

[`examples/class.yaml`](examples/class.yaml) is the same likelihoods.
`classy` has `requires: [YHe, N_ur]`, with `N_ur = Neff_BBN - nur_shift`
and one massive neutrino (`N_ncdm: 1`, `m_ncdm: 0.06`, `T_ncdm: 0.71611`).

## Checks

```bash
JAX_PLATFORMS=cpu \
LINX_PATH=/path/to/LINX \
LINX_COBAYA_CHECK_YAML=/path/to/test_linx.yaml \
python linx_cobaya/check_linx_cobaya.py
```

The yaml has to be a Cobaya run with `LinxBBN` and `camb`. The script checks
that CAMB's `YHe` and `N_eff` match LINX. The first call compiles LINX and
takes about a minute.

## What the code fixes

`dNeff_convention` defaults to `"init"`: the sampled `dNeff` is LINX's native
ΔN_eff at the start of the integration (T ≈ 8.6 MeV), passed through directly.
Set `dNeff_convention: final` to sample the post-annihilation ΔN_eff instead;
the wrapper converts it using a slope calibrated at startup. Either way
`Neff_BBN` is the value LINX produced and the Boltzmann code received.

`Y_p` from LINX is the helium nucleon fraction. `YHe` is the mass fraction,
converted with `camb.bbn.ypBBN_to_yhe`.

Non-finite predictions make `calculate` return `False` (zero likelihood).

`warmup: true` runs one LINX solve in `initialize()` so the XLA compile happens
at startup. With `dNeff_convention: "final"` that solve runs anyway, because
the slope calibration needs a compiled model.

`sample_nuclear` defaults to `true`. Cobaya automatically adds `tau_n_fac`
with a Gaussian prior of mean 1 and standard deviation 0.000682, and one
`q_<reaction>` per reaction in the selected `nuclear_net`, each with a
standard normal prior. You can override these priors, reference values, or
proposal widths in the run's `params` block, as with Planck nuisances.
Set `sample_nuclear: false` to keep `tau_n_fac=1` and all reaction shifts at zero.

The prior widths follow `LINX/scripts/CMB_BBN_marg_nuisance_omegab_Neff.py`.
The `ref` and `proposal` widths are sampler tuning.

`Y_p` and `D/H` use species indices looked up by name in LINX's `species_dict`.
Helium-4 is `"a"`.

## Citing

This package is MIT-licensed, but it is research software: please cite the
underlying codes, which did the actual physics.

**Always:**

- **LINX** — Giovanetti, Lisanti, Liu, Mishra-Sharma & Ruderman,
  [Phys. Rev. D 112, 063505 (2025)](https://doi.org/10.1103/f3tj-r882),
  [arXiv:2408.14538](https://arxiv.org/abs/2408.14538)
- **Cobaya** — Torrado & Lewis,
  [JCAP 05 (2021) 057](https://doi.org/10.1088/1475-7516/2021/05/057),
  [arXiv:2005.05290](https://arxiv.org/abs/2005.05290)

**Boltzmann codes:**

- **CAMB** — Lewis, Challinor & Lasenby,
  [ApJ 538, 473 (2000)](https://doi.org/10.1086/309179),
  [astro-ph/9911177](https://arxiv.org/abs/astro-ph/9911177)
- **CLASS** — Blas, Lesgourgues & Tram,
  [JCAP 07 (2011) 034](https://doi.org/10.1088/1475-7516/2011/07/034),
  [arXiv:1104.2933](https://arxiv.org/abs/1104.2933)

**If you use the default abundance measurements in `BBNAbundances`:**

- **D/H** — Cooke, Pettini & Steidel,
  [ApJ 855, 102 (2018)](https://doi.org/10.3847/1538-4357/aaab53)
- **Y_p** — Aver et al., *The LBT Y_p Project IV*,
  [arXiv:2601.22238](https://arxiv.org/abs/2601.22238)

A machine-readable version of this list is in
[`CITATION.cff`](CITATION.cff).
