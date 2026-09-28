"""
Verification suite for the LINX <-> Cobaya coupling.

The Boltzmann-solver checks use CAMB. Point it at a LINX checkout and a Cobaya
yaml containing LinxBBN plus camb:

    JAX_PLATFORMS=cpu \\
    LINX_PATH=/path/to/LINX \\
    LINX_COBAYA_CHECK_YAML=/path/to/test_linx.yaml \\
    python linx_cobaya/check_linx_cobaya.py

Each check prints PASS/FAIL; the script exits nonzero if any check fails.
"""

import os
import sys
import time

import numpy as np

# Repository root: the directory containing the `linx_cobaya` package.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

LINX_PATH = os.environ.get("LINX_PATH", os.path.join(PACKAGE_ROOT, "LINX"))
TEST_YAML = os.environ.get(
    "LINX_COBAYA_CHECK_YAML",
    os.path.join(PACKAGE_ROOT, "examples", "test_linx.yaml"),
)
if not os.path.isdir(os.path.join(LINX_PATH, "linx")):
    sys.exit(
        f"LINX checkout not found at {LINX_PATH} (no 'linx' package inside).\n"
        "Set LINX_PATH to the root of your LINX checkout."
    )
if not os.path.exists(TEST_YAML):
    sys.exit(
        f"Check config not found at {TEST_YAML}.\n"
        "Set LINX_COBAYA_CHECK_YAML to a Cobaya yaml containing LinxBBN and camb."
    )

FIDUCIAL = {"ombh2": 0.02242, "dNeff": 0.0}

_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"\n       {detail}" if detail else ""))


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# --------------------------------------------------------------------------
section("1. LINX standalone reproduces the SBBN reference")
# --------------------------------------------------------------------------
# Reference values from LINX/scripts/test_SBBN_PRIMAT.py:74-78 (key_PRIMAT_2023).
REF = {"Neff": 3.0435397, "Y_p": 0.24711426, "DH_e5": 2.4379026}

from linx_cobaya import BBNAbundances, LinxBBN  # noqa: E402

t0 = time.time()
linx = LinxBBN({"linx_path": LINX_PATH, "warmup": True})
print(f"       (initialize + JIT compile took {time.time() - t0:.1f} s)")

res = linx._compute(FIDUCIAL["ombh2"], FIDUCIAL["dNeff"])
got = {"Neff": res["Neff_BBN"], "Y_p": res["Y_p"], "DH_e5": res["DH"] * 1e5}
pct = {k: 100 * (got[k] - REF[k]) / REF[k] for k in REF}
check(
    "SBBN values within 0.3% of the PRIMAT reference",
    all(abs(v) < 0.3 for v in pct.values()),
    ", ".join(f"{k}={got[k]:.6f} ({pct[k]:+.4f}%)" for k in REF),
)

# --------------------------------------------------------------------------
section("2. Wrapper agrees with CAMB's own PRIMAT table")
# --------------------------------------------------------------------------
# Both are PRIMAT-based, so YHe should agree closely. A large discrepancy means
# the nucleon/mass-fraction conversion or eta_fac is wrong -- this is the single
# most valuable check here.
from camb.bbn import BBN_table_interpolator  # noqa: E402

table = BBN_table_interpolator()
table_YHe = float(table.Y_He(FIDUCIAL["ombh2"], 0.0))
table_Yp = float(table.Y_p(FIDUCIAL["ombh2"], 0.0))
table_DH = float(table.DH(FIDUCIAL["ombh2"], 0.0))

d_YHe = abs(res["YHe"] - table_YHe) / table_YHe
d_Yp = abs(res["Y_p"] - table_Yp) / table_Yp
d_DH = abs(res["DH"] - table_DH) / table_DH
check(
    "YHe agrees with CAMB's PRIMAT table to better than 1%",
    d_YHe < 0.01,
    f"LINX YHe={res['YHe']:.6f} vs table {table_YHe:.6f} ({100 * d_YHe:.3f}%); "
    f"Y_p {res['Y_p']:.6f} vs {table_Yp:.6f} ({100 * d_Yp:.3f}%); "
    f"D/H {res['DH']:.4e} vs {table_DH:.4e} ({100 * d_DH:.3f}%)",
)
from camb.bbn import yhe_to_ypBBN  # noqa: E402

check(
    "YHe (mass) and Y_p (nucleon) round-trip through CAMB's conversion",
    res["YHe"] < res["Y_p"] and abs(yhe_to_ypBBN(res["YHe"]) - res["Y_p"]) < 1e-12,
    f"YHe={res['YHe']:.8f} (mass), Y_p={res['Y_p']:.8f} (nucleon), "
    f"yhe_to_ypBBN(YHe)={yhe_to_ypBBN(res['YHe']):.8f}",
)

# --------------------------------------------------------------------------
section("3. Cobaya wiring: LINX -> CAMB")
# --------------------------------------------------------------------------
from cobaya.model import get_model  # noqa: E402
from cobaya.yaml import yaml_load_file  # noqa: E402

info = yaml_load_file(TEST_YAML)
info.pop("sampler", None)
model = get_model(info)
# The BBN likelihood alone needs nothing from CAMB, so ask for a spectrum
# explicitly -- otherwise cobaya never sets up the CAMB transfers.
model.add_requirements({"Cl": {"tt": 2500}, "CAMBdata": None})

point = dict(FIDUCIAL)
logpost = model.logposterior(point)
derived = dict(zip(model.parameterization.derived_params(), logpost.derived))

check("logposterior is finite", np.isfinite(logpost.logpost), f"logpost={logpost.logpost:.4f}")
check(
    "all LINX products present as derived params",
    all(k in derived for k in ("YHe", "Y_p", "DH", "nnu", "Neff_BBN", "DHBBN")),
    ", ".join(f"{k}={derived[k]:.6g}" for k in ("YHe", "Y_p", "DH", "nnu", "Neff_BBN")),
)

# The decisive check: did CAMB actually *consume* LINX's values, or silently
# fall back to its own BBN table?
camb_params = model.provider.get_CAMBdata().Params
check(
    "CAMB consumed LINX's YHe",
    abs(camb_params.YHe - derived["YHe"]) < 1e-12,
    f"CAMBparams.YHe={camb_params.YHe:.10f} vs LinxBBN YHe={derived['YHe']:.10f} "
    f"(CAMB's own table would give {table_YHe:.10f})",
)
check(
    "CAMB consumed LINX's nnu",
    abs(camb_params.N_eff - derived["Neff_BBN"]) < 1e-9,
    f"CAMBparams.N_eff={camb_params.N_eff:.10f} vs Neff_BBN={derived['Neff_BBN']:.10f}",
)
check(
    "Neff_BBN is LINX's own final N_eff, not 3.044 + dNeff",
    abs(derived["Neff_BBN"] - res["Neff_BBN"]) < 1e-9
    and abs(derived["Neff_BBN"] - (3.044 + point["dNeff"])) > 1e-4,
    f"Neff_BBN={derived['Neff_BBN']:.7f} (LINX standalone gave {res['Neff_BBN']:.7f}); "
    f"3.044+dNeff would be {3.044 + point['dNeff']:.7f}",
)

# --------------------------------------------------------------------------
section("4. Timing")
# --------------------------------------------------------------------------
n = 5
t0 = time.time()
for i in range(n):
    linx._compute(0.0222 + 1e-5 * i, 0.01 * i)
t_linx = (time.time() - t0) / n

t0 = time.time()
for i in range(n):
    model.logposterior({"ombh2": 0.0222 + 1e-5 * i, "dNeff": 0.01 * i})
t_full = (time.time() - t0) / n

check(
    "LINX stays compiled across varying parameter values",
    t_linx < 10.0,
    f"LINX alone {t_linx:.2f} s/call; full logposterior {t_full:.2f} s/call "
    f"(CAMB share ~{t_full - t_linx:.2f} s). If LINX is not far below the first "
    "call's ~50 s, something is re-triggering JIT.",
)

# --------------------------------------------------------------------------
section("5. dNeff propagates through to CAMB and the spectra")
# --------------------------------------------------------------------------
lp0 = model.logposterior({"ombh2": 0.02242, "dNeff": 0.0})
d0 = dict(zip(model.parameterization.derived_params(), lp0.derived))
cl0 = model.provider.get_Cl(ell_factor=True)["tt"][2:2500].copy()

lp3 = model.logposterior({"ombh2": 0.02242, "dNeff": 0.3})
d3 = dict(zip(model.parameterization.derived_params(), lp3.derived))
cl3 = model.provider.get_Cl(ell_factor=True)["tt"][2:2500].copy()

check(
    "Neff_BBN tracks dNeff one-for-one under dNeff_convention='final'",
    abs((d3["Neff_BBN"] - d0["Neff_BBN"]) - 0.3) < 2e-3,
    f"Neff_BBN: {d0['Neff_BBN']:.6f} -> {d3['Neff_BBN']:.6f} "
    f"(delta {d3['Neff_BBN'] - d0['Neff_BBN']:.6f}, requested 0.3)",
)
check(
    "Y_p responds to dNeff",
    abs(d3["Y_p"] - d0["Y_p"]) > 1e-3,
    f"Y_p: {d0['Y_p']:.6f} -> {d3['Y_p']:.6f}",
)
check(
    "TT spectrum responds to dNeff",
    np.max(np.abs(cl3 / cl0 - 1)) > 1e-3,
    f"max |dCl/Cl| = {np.max(np.abs(cl3 / cl0 - 1)):.4f}",
)

# --------------------------------------------------------------------------
section("5b. The dNeff convention conversion is what it claims")
# --------------------------------------------------------------------------
# 'init' passes LINX's native pre-e+e- parameter straight through, where the
# same numeric value produces only ~0.272x the change in final N_eff.
linx_init = LinxBBN(
    {"linx_path": LINX_PATH, "warmup": False, "dNeff_convention": "init"}
)
r_init_0 = linx_init._compute(0.02242, 0.0)
r_init_3 = linx_init._compute(0.02242, 0.3)
ratio = (r_init_3["Neff_BBN"] - r_init_0["Neff_BBN"]) / 0.3
check(
    "dNeff_convention='init' shows the raw ~0.272 dilution",
    abs(ratio - 0.2718) < 5e-3,
    f"native dNeff=0.3 -> delta N_eff = {r_init_3['Neff_BBN'] - r_init_0['Neff_BBN']:.6f} "
    f"(ratio {ratio:.5f}); 'final' convention corrects for exactly this",
)
check(
    "calibrated slope matches the measured dilution",
    abs(linx._dNeff_slope - ratio) < 5e-3,
    f"LinxBBN calibrated slope = {linx._dNeff_slope:.5f}, measured = {ratio:.5f}",
)

# --------------------------------------------------------------------------
section("6. Robustness at extreme parameter values")
# --------------------------------------------------------------------------
# stop_at_error is True in the test yaml; rebuild with it off so a failure is
# converted to -inf rather than an exception, which is the MCMC behaviour.
info_soft = yaml_load_file(TEST_YAML)
info_soft.pop("sampler", None)
info_soft["theory"]["linx_cobaya.linx_bbn.LinxBBN"]["stop_at_error"] = False
info_soft["theory"]["linx_cobaya.linx_bbn.LinxBBN"]["warmup"] = False
info_soft["theory"]["camb"]["stop_at_error"] = False
model_soft = get_model(info_soft)
model_soft.add_requirements({"Cl": {"tt": 2500}, "CAMBdata": None})
# Extreme but not unphysical: LINX converges here, so the correct behaviour is
# a large chi2, not -inf. What matters is that nothing raises.
bad_points = [
    {"ombh2": 0.09, "dNeff": 0.0},
    {"ombh2": 0.005, "dNeff": 0.0},
    {"ombh2": 0.02242, "dNeff": 0.99},
    {"ombh2": 0.02242, "dNeff": -0.99},
]
ok, detail = True, []
for p in bad_points:
    try:
        lp = model_soft.logposterior(p)
        detail.append(f"ombh2={p['ombh2']}, dNeff={p['dNeff']} -> logpost={lp.logpost:.4g}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        detail.append(f"ombh2={p['ombh2']}, dNeff={p['dNeff']} -> RAISED {exc!r}")
check(
    "extreme parameters return a value (finite or -inf) without raising",
    ok,
    "; ".join(detail),
)

# --------------------------------------------------------------------------
section("Summary")
# --------------------------------------------------------------------------
n_fail = sum(1 for _, ok in _results if not ok)
for name, ok in _results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
sys.exit(1 if n_fail else 0)
