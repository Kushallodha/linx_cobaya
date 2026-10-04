"""
Verification suite for the LINX <-> Cobaya coupling.

The Boltzmann-solver checks use CAMB. Point it at a LINX checkout and a Cobaya
yaml containing LinxBBN plus camb:

    JAX_PLATFORMS=cpu \\
    LINX_PATH=/path/to/LINX \\
    LINX_COBAYA_CHECK_YAML=/path/to/camb.yaml \\
    python linx_cobaya/check_linx_cobaya.py

Section 7 needs a CLASS config and is skipped without one:

    LINX_COBAYA_CHECK_CLASS_YAML=/path/to/class.yaml

Each check prints PASS/FAIL; the script exits nonzero if any check fails.
"""

import copy
import os
import sys
import time

import numpy as np
from cobaya.model import get_model
from cobaya.tools import get_scipy_1d_pdf
from cobaya.yaml import yaml_load_file

# Repository root: the directory containing the `linx_cobaya` package.
PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_ROOT)

LINX_PATH = os.environ.get("LINX_PATH", os.path.join(PACKAGE_ROOT, "LINX"))
TEST_YAML = os.environ.get(
    "LINX_COBAYA_CHECK_YAML",
    os.path.join(PACKAGE_ROOT, "examples", "camb.yaml"),
)
CLASS_YAML = os.environ.get(
    "LINX_COBAYA_CHECK_CLASS_YAML",
    os.path.join(PACKAGE_ROOT, "examples", "class.yaml"),
)
OMBH2_SCAN = (0.021, 0.02238, 0.024)

FIDUCIAL = {
    "ombh2": 0.02242, "dNeff": 0.0, "tau_n_fac": 1.0,
    "logA": 3.0428, "ns": 0.9695, "omegam": 0.3017,
    "H0": 68.38, "tau": 0.05672,
}


def complete_point(model, point):
    """Fill sampled params from `point`, then FIDUCIAL, then reference medians."""
    definitions = model.info()["params"]
    result = {}
    for name in model.parameterization.sampled_params():
        if name in point:
            value = point[name]
        elif name in FIDUCIAL:
            value = FIDUCIAL[name]
        else:
            definition = definitions[name]
            reference = definition.get("ref")
            if reference is None:
                reference = definition["prior"]
            value = (
                get_scipy_1d_pdf(reference).ppf(0.5)
                if isinstance(reference, dict) else reference
            )
        if not np.isfinite(value):
            raise ValueError(
                f"No finite fiducial value for {name!r}; supply it explicitly."
            )
        result[name] = float(value)
    return result


_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(
        f"[{'PASS' if ok else 'FAIL'}] {name}"
        + (f"\n       {detail}" if detail else "")
    )


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def class_scan(cfg):
    """Build once and scan ombh2; return CLASS's inputs and theta_s_100 per value.

    LinxBBN's compile and the Planck data load are ~1 min each, so ombh2 is
    varied by re-evaluating one model rather than rebuilding it.
    """
    model = get_model(copy.deepcopy(cfg))
    base = complete_point(model, {"ombh2": OMBH2_SCAN[1], "dNeff": 0.3})
    pars, thetas = {}, []
    for ombh2 in OMBH2_SCAN:
        model.logposterior({**base, "ombh2": ombh2})
        classy = model.theory["classy"].classy
        thetas.append(
            float(classy.get_current_derived_parameters(["theta_s_100"])["theta_s_100"])
        )
        if ombh2 == OMBH2_SCAN[1]:
            pars = dict(classy.pars)
    return pars, thetas


def run_class_checks():
    # classy is agnostic: Cobaya hands it only the input params no other
    # component claimed. LinxBBN claims ombh2, so CLASS would silently fall back
    # to its own default baryon density. examples/class.yaml works around this by
    # defining omega_b = ombh2, which LinxBBN does not claim.
    info = yaml_load_file(CLASS_YAML)
    info.pop("sampler", None)
    info.pop("output", None)
    if "/path/to" in str(info):
        print(f"[SKIP] CLASS checks: edit the /path/to placeholders in {CLASS_YAML}")
        return

    pars, thetas = class_scan(info)
    om = pars.get("Omega_m")
    check(
        "CLASS is given Omega_m and derives omega_cdm itself",
        om is not None and abs(float(om) - 0.3017) < 1e-9 and "omega_cdm" not in pars,
        f"Omega_m={om!r}, omega_cdm passed: {'omega_cdm' in pars}",
    )
    # Names CLASS does not understand, or that cobaya should have renamed.
    stale = [
        k for k in ("ombh2", "omch2", "omegam", "tau", "ns", "As", "mnu")
        if k in pars
    ]
    check("no unrenamed parameter names reach CLASS", not stale, f"passed: {stale}")

    # cobaya#494's own diagnostic: if the density never arrives, a quantity
    # that depends on it is frozen against changes in the sampled value.
    spread = max(thetas) - min(thetas)
    check(
        "theta_s_100 responds to ombh2",
        spread > 1e-4 * abs(np.mean(thetas)),
        f"theta_s_100 = {['%.6g' % t for t in thetas]} (spread {spread:.3e})",
    )

    # Remove omega_b as a control; residual dependence can remain through YHe.
    broken = copy.deepcopy(info)
    broken["params"].pop("omega_b", None)
    try:
        pars_b, thetas_b = class_scan(broken)
        spread_b = max(thetas_b) - min(thetas_b)
        check(
            "control: without omega_b, CLASS never receives the baryon density",
            pars_b.get("omega_b") is None,
            f"omega_b={pars_b.get('omega_b')!r} (CLASS falls back to its own default)",
        )
        check(
            "control: without omega_b, the theta_s_100 response is suppressed",
            spread_b < spread / 2,
            f"spread {spread_b:.3e} vs {spread:.3e} with the fix "
            f"({100 * spread_b / spread:.1f}%); any residue is ombh2 reaching "
            "CLASS indirectly through LINX's YHe",
        )
    except Exception as exc:  # noqa: BLE001
        check("control: without omega_b, CLASS falls back", False, f"raised {exc!r}")


def main():
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

    # --------------------------------------------------------------------------
    section("1. LINX standalone reproduces the SBBN reference")
    # --------------------------------------------------------------------------
    # Reference values from LINX/scripts/test_SBBN_PRIMAT.py:74-78 (key_PRIMAT_2023).
    REF = {"Neff": 3.0435397, "Y_p": 0.24711426, "DH_e5": 2.4379026}

    from linx_cobaya import BBNAbundances, LinxBBN

    t0 = time.time()
    # dNeff_convention='final' so the slope is calibrated for section 5b. At
    # dNeff=0 the conversion is the identity, so sections 1-4 are unaffected.
    linx = LinxBBN(
        {"linx_path": LINX_PATH, "warmup": True, "dNeff_convention": "final"}
    )
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
    # A large YHe mismatch means the Y_p -> YHe conversion or eta_fac is wrong.
    from camb.bbn import BBN_table_interpolator, yhe_to_ypBBN

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
    check(
        "YHe (mass) and Y_p (nucleon) round-trip through CAMB's conversion",
        res["YHe"] < res["Y_p"] and abs(yhe_to_ypBBN(res["YHe"]) - res["Y_p"]) < 1e-12,
        f"YHe={res['YHe']:.8f} (mass), Y_p={res['Y_p']:.8f} (nucleon), "
        f"yhe_to_ypBBN(YHe)={yhe_to_ypBBN(res['YHe']):.8f}",
    )

    # --------------------------------------------------------------------------
    section("3. Cobaya wiring: LINX -> CAMB")
    # --------------------------------------------------------------------------
    info = yaml_load_file(TEST_YAML)
    info.pop("sampler", None)
    model = get_model(info)
    # The BBN likelihood alone needs nothing from CAMB, so ask for a spectrum
    # explicitly -- otherwise cobaya never sets up the CAMB transfers.
    model.add_requirements({"Cl": {"tt": 2500}, "CAMBdata": None})

    logpost = model.logposterior(complete_point(model, FIDUCIAL))
    derived = dict(zip(model.parameterization.derived_params(), logpost.derived))

    check(
        "logposterior is finite",
        np.isfinite(logpost.logpost),
        f"logpost={logpost.logpost:.4f}",
    )
    check(
        "all LINX products present as derived params",
        all(k in derived for k in ("YHe", "Y_p", "DH", "nnu", "Neff_BBN", "DHBBN")),
        ", ".join(
            f"{k}={derived[k]:.6g}" for k in ("YHe", "Y_p", "DH", "nnu", "Neff_BBN")
        ),
    )

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
        f"CAMBparams.N_eff={camb_params.N_eff:.10f} "
        f"vs Neff_BBN={derived['Neff_BBN']:.10f}",
    )
    check(
        "Neff_BBN is LINX's own final N_eff, not 3.044 + dNeff",
        abs(derived["Neff_BBN"] - res["Neff_BBN"]) < 1e-9
        and abs(derived["Neff_BBN"] - (3.044 + FIDUCIAL["dNeff"])) > 1e-4,
        f"Neff_BBN={derived['Neff_BBN']:.7f} "
        f"(LINX standalone gave {res['Neff_BBN']:.7f}); "
        f"3.044+dNeff would be {3.044 + FIDUCIAL['dNeff']:.7f}",
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
        point = {"ombh2": 0.0222 + 1e-5 * i, "dNeff": 0.01 * i}
        model.logposterior(complete_point(model, point))
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
    lp0 = model.logposterior(complete_point(model, {"ombh2": 0.02242, "dNeff": 0.0}))
    d0 = dict(zip(model.parameterization.derived_params(), lp0.derived))
    cl0 = model.provider.get_Cl(ell_factor=True)["tt"][2:2500].copy()

    lp3 = model.logposterior(complete_point(model, {"ombh2": 0.02242, "dNeff": 0.3}))
    d3 = dict(zip(model.parameterization.derived_params(), lp3.derived))
    cl3 = model.provider.get_Cl(ell_factor=True)["tt"][2:2500].copy()

    # What Neff_BBN is expected to do depends on which convention the yaml picked.
    convention = model.theory["linx_cobaya.linx_bbn.LinxBBN"].dNeff_convention
    d_neff = d3["Neff_BBN"] - d0["Neff_BBN"]
    if convention == "final":
        check(
            "Neff_BBN tracks dNeff one-for-one under dNeff_convention='final'",
            abs(d_neff - 0.3) < 2e-3,
            f"Neff_BBN: {d0['Neff_BBN']:.6f} -> {d3['Neff_BBN']:.6f} "
            f"(delta {d_neff:.6f}, requested 0.3)",
        )
    else:
        check(
            "Neff_BBN responds to dNeff, diluted, under dNeff_convention='init'",
            0.0 < d_neff < 0.3,
            f"Neff_BBN: {d0['Neff_BBN']:.6f} -> {d3['Neff_BBN']:.6f} "
            f"(delta {d_neff:.6f} for a requested pre-e+e- 0.3)",
        )
    # Y_p rises by ~0.013 per unit N_eff; require at least ~40% of that.
    check(
        "Y_p responds to dNeff",
        abs(d3["Y_p"] - d0["Y_p"]) > 5e-3 * d_neff,
        f"Y_p: {d0['Y_p']:.6f} -> {d3['Y_p']:.6f} "
        f"(delta {d3['Y_p'] - d0['Y_p']:+.6f} for a final Neff shift of {d_neff:.6f})",
    )
    check(
        "TT spectrum responds to dNeff",
        np.max(np.abs(cl3 / cl0 - 1)) > 1e-3,
        f"max |dCl/Cl| = {np.max(np.abs(cl3 / cl0 - 1)):.4f}",
    )

    # --------------------------------------------------------------------------
    section("5b. The dNeff convention conversion is what it claims")
    # --------------------------------------------------------------------------
    # 'init' passes LINX's native pre-e+e- parameter straight through, so the same
    # numeric value produces a smaller change in final N_eff.
    linx_init = LinxBBN(
        {"linx_path": LINX_PATH, "warmup": False, "dNeff_convention": "init"}
    )
    r_init_0 = linx_init._compute(0.02242, 0.0)
    r_init_3 = linx_init._compute(0.02242, 0.3)
    delta_init = r_init_3["Neff_BBN"] - r_init_0["Neff_BBN"]
    ratio = delta_init / 0.3
    check(
        "dNeff_convention='init' dilutes: final N_eff moves by less than the input",
        0.0 < ratio < 1.0,
        f"native dNeff=0.3 -> delta N_eff = {delta_init:.6f} "
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
    # Evaluate with stop_at_error=False to check rejection of failed points.
    info_soft = yaml_load_file(TEST_YAML)
    info_soft.pop("sampler", None)
    info_soft["theory"]["linx_cobaya.linx_bbn.LinxBBN"]["warmup"] = False
    for opts in info_soft["theory"].values():
        if isinstance(opts, dict):
            opts["stop_at_error"] = False
    model_soft = get_model(info_soft)
    model_soft.add_requirements({"Cl": {"tt": 2500}, "CAMBdata": None})
    # Edges of the yaml's own priors: outside them cobaya returns -inf without
    # ever calling LINX, which would test nothing.
    bounds = dict(zip(model_soft.parameterization.sampled_params(),
                      model_soft.prior.bounds()))
    bad_points = [
        {"ombh2": bounds["ombh2"][0], "dNeff": 0.0},
        {"ombh2": bounds["ombh2"][1], "dNeff": 0.0},
        {"ombh2": FIDUCIAL["ombh2"], "dNeff": bounds["dNeff"][0]},
        {"ombh2": FIDUCIAL["ombh2"], "dNeff": bounds["dNeff"][1]},
    ]
    ok, detail = True, []
    for p in bad_points:
        where = f"ombh2={p['ombh2']}, dNeff={p['dNeff']}"
        try:
            lp = model_soft.logposterior(complete_point(model_soft, p))
            detail.append(f"{where} -> logpost={lp.logpost:.4g}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail.append(f"{where} -> RAISED {exc!r}")
    check(
        "extreme parameters return a value (finite or -inf) without raising",
        ok,
        "; ".join(detail),
    )

    # --------------------------------------------------------------------------
    section("7. CLASS receives the sampled baryon density (CobayaSampler/cobaya#494)")
    # --------------------------------------------------------------------------
    if not os.path.exists(CLASS_YAML):
        print(f"[SKIP] CLASS checks: no CLASS yaml at {CLASS_YAML}")
    else:
        try:
            import classy  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] CLASS checks: classy not importable ({exc})")
        else:
            run_class_checks()

    # --------------------------------------------------------------------------
    section("Summary")
    # --------------------------------------------------------------------------
    n_fail = sum(1 for _, ok in _results if not ok)
    for name, ok in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
