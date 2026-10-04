"""
Cobaya Theory component wrapping LINX (https://arxiv.org/abs/2408.14538).

Replaces a Boltzmann code's static BBN interpolation table with a live BBN
network solve. At each step LINX is given (ombh2, dNeff) and returns YHe and
the final N_eff, which are handed to CAMB or CLASS as *input* parameters, plus
Y_p and D/H as derived parameters for a BBN abundance likelihood.

See linx_cobaya/README.md for the YAML wiring.
"""

import contextlib
import io
import os
import sys

import numpy as np

from cobaya.theory import Theory

# LINX SM reference values (key_PRIMAT_2023, ombh2 = 0.02242, dNeff = 0),
# used only for the warm-up call.
FIDUCIAL_OMBH2 = 0.02242


def _add_linx_path(linx_path):
    if linx_path:
        linx_path = os.path.abspath(os.path.expanduser(linx_path))
        if not os.path.isdir(os.path.join(linx_path, "linx")):
            raise ValueError(
                f"linx_path={linx_path!r} does not contain a 'linx' package "
                "directory. It should point at the root of the LINX repository."
            )
        if linx_path not in sys.path:
            sys.path.insert(0, linx_path)


class LinxBBN(Theory):
    """
    BBN abundances from LINX, provided to CAMB or CLASS and to likelihoods.

    Provides
    --------
    YHe : helium *mass* fraction, the ``YHe`` input of both CAMB and CLASS.
    Y_p : helium *nucleon* fraction 4*n_He4/n_b, CAMB's ``Y_p`` convention.
    DH  : deuterium number ratio n_D/n_H.
    nnu : final N_eff, CAMB's ``nnu`` input (only if ``provide_nnu``).
    N_ur : Neff_BBN - ``nur_shift``, CLASS's ultra-relativistic input. Always
        provided; a CAMB run simply does not require it.
    Neff_BBN : the final N_eff, always provided, so it is recorded even when
        ``provide_nnu`` is False.

    Inputs
    ------
    ombh2 : baryon density; enters LINX only as eta_fac = ombh2 / 0.02242.
    dNeff : extra effective number of neutrino species. Its meaning is set by
        ``dNeff_convention``; see below.
    tau_n_fac, q_<reaction> : optional nuclear nuisances when
        ``sample_nuclear`` is True. LinxBBN adds default priors for these, one
        q per reaction in ``nuclear_net``: tau_n_fac ~ N(1, 0.000682) and each
        q ~ N(0, 1). Override them in the run's params block.

    The dNeff convention
    --------------------
    LINX's native input ``Delt_Neff_init`` is Delta N_eff at the *start* of the
    integration (T ~ 8.6 MeV), i.e. before e+e- annihilation. The extra
    radiation then redshifts as a^-4 while the photons are heated by e+e-
    annihilation, so the final Delta N_eff is smaller than the value passed in
    by the entropy-transfer factor.

    This class defaults to ``dNeff_convention="init"``: the sampled ``dNeff``
    is that native pre-e+e- parameter, passed straight through.

    Set ``dNeff_convention="final"`` to sample Delta N_eff in the post-e+e-
    sense the CMB literature uses instead. The class then converts to LINX's
    native input by dividing by a slope ``d(Neff_final) / d(Delt_Neff_init)``.
    The slope is measured at startup from background solves at
    ``Delt_Neff_init`` = 0 and 1.

    Either way ``Neff_BBN`` reports the value LINX actually produced and that
    the Boltzmann code actually receives, so the two codes are never
    inconsistent -- the convention only relabels the sampled coordinate.

    Note that ``dNeff_convention="final"`` forces the warm-up solve regardless
    of ``warmup``, because the calibration needs a compiled model.
    """

    # --- YAML-settable options -------------------------------------------
    linx_path: str = ""
    nuclear_net: str = "key_PRIMAT_2023"
    rtol: float = 1e-6
    atol: float = 1e-9
    sampling_nTOp: int = 150
    max_steps: int = 4096
    bkg_max_steps: int = 512
    provide_nnu: bool = True
    # N_ur = Neff_BBN - nur_shift. 1.0132 is one massive neutrino at
    # T_ncdm = 0.71611 (CLASS explanatory.ini).
    nur_shift: float = 1.0132
    warmup: bool = True
    dNeff_convention: str = "init"
    # If True, sample tau_n_fac and one q_<reaction> per network rate, the same
    # marginalisation as LINX/scripts/CMB_BBN_marg_nuisance_omegab_Neff.py.
    sample_nuclear: bool = True

    @classmethod
    def _network_q_params(cls, nuclear_net, linx_path):
        _add_linx_path(linx_path)
        from linx.nuclear import NuclearRates

        rates = NuclearRates(nuclear_net=nuclear_net)
        return [f"q_{rxn.name}" for rxn in rates.reactions]

    @classmethod
    def get_modified_defaults(cls, defaults, input_options=None):
        """Supply network nuisance priors before Cobaya builds its parameterization."""
        options = {**defaults, **(input_options or {})}
        if not options["sample_nuclear"]:
            return defaults
        q_names = cls._network_q_params(options["nuclear_net"], options["linx_path"])
        params = defaults.setdefault("params", {})
        params["tau_n_fac"] = {
            "prior": {"dist": "norm", "loc": 1.0, "scale": 0.000682},
            "ref": {"dist": "norm", "loc": 1.0, "scale": 0.00013},
            "proposal": 0.00033,
            "latex": r"\tau_n / \tau_n^{(0)}",
        }
        for name in q_names:
            params[name] = {
                "prior": {"dist": "norm", "loc": 0.0, "scale": 1.0},
                "ref": {"dist": "norm", "loc": 0.0, "scale": 0.2},
                "proposal": 0.5,
                "latex": r"q_\mathrm{" + name[2:] + "}",
            }
        return defaults

    def initialize(self):
        if self.dNeff_convention not in ("final", "init"):
            raise ValueError(
                f"dNeff_convention must be 'final' or 'init', "
                f"not {self.dNeff_convention!r}."
            )
        _add_linx_path(self.linx_path)

        # NB: importing any LINX module enables JAX x64 globally (linx/const.py:3).
        import jax.numpy as jnp

        import linx
        import linx.const as const
        from linx.abundances import AbundanceModel
        from linx.background import BackgroundModel
        from linx.nuclear import NuclearRates

        # CAMB's own nucleon-fraction -> mass-fraction conversion, so that the
        # convention matches exactly what CAMB would have used internally.
        from camb.bbn import ypBBN_to_yhe

        self._jnp = jnp
        self._linx = linx
        self._ypBBN_to_yhe = ypBBN_to_yhe

        # throw=False: a failed solve returns NaN, which calculate() turns into
        # zero likelihood instead of killing the chain.
        self._bkg = BackgroundModel(throw=False, max_steps=self.bkg_max_steps)
        self._abd = AbundanceModel(
            NuclearRates(nuclear_net=self.nuclear_net), throw=False
        )

        # Fixed-shape zero nuisance vector when not sampling. Always pass an
        # array (not None) so the jitted AbundanceModel path stays stable.
        self._q_param_names = [
            f"q_{rxn.name}" for rxn in self._abd.nuclear_net.reactions
        ]
        self._n_reactions = len(self._q_param_names)
        self._q0 = jnp.zeros(self._n_reactions)

        # Indices from species_dict. Helium-4 is "a".
        species_by_name = {
            name: i for i, name in self._abd.species_dict.items()
        }
        try:
            self._i_p = species_by_name["p"]
            self._i_d = species_by_name["d"]
            self._i_He4 = species_by_name["a"]
        except KeyError as exc:
            raise RuntimeError(
                f"LINX species_dict has no entry {exc.args[0]!r}; cannot locate "
                f"the species needed for Y_p and D/H. Got {self._abd.species_dict}."
            ) from exc
        n_species = self._abd.nuclear_net.max_i_species
        if max(self._i_p, self._i_d, self._i_He4) >= n_species:
            raise RuntimeError(
                f"nuclear_net={self.nuclear_net!r} tracks only {n_species} species, "
                f"which does not include all of p={self._i_p}, d={self._i_d}, "
                f"He4={self._i_He4}. Y_p and D/H cannot be computed."
            )

        # ombh2 -> eta_fac reference, see const.py:109-114.
        self._ombh2_ref = float(const.Omegabh2)

        self.log.info(
            "LINX network %r: %d reactions, %d species.%s",
            self.nuclear_net,
            self._n_reactions,
            n_species,
            (
                " Sampling tau_n_fac + " + ", ".join(self._q_param_names) + "."
                if self.sample_nuclear
                else " Nuclear nuisances fixed (tau_n_fac=1, q=0)."
            ),
        )

        # Calibration of Delt_Neff_init -> final N_eff, filled in below.
        # _dNeff_slope is d(Neff_final) / d(Delt_Neff_init).
        self._Neff_sm = None
        self._dNeff_slope = 1.0

        if not self.warmup and self.dNeff_convention == "final":
            self.log.info(
                "warmup=False ignored: dNeff_convention='final' needs a compiled "
                "model to calibrate the Delt_Neff_init -> N_eff slope. Use "
                "dNeff_convention='init' if you need to skip the startup compile."
            )

        if self.warmup or self.dNeff_convention == "final":
            # Compile at startup and suppress LINX's trace-time banner.
            self.log.info("Compiling LINX (this takes ~1 minute)...")
            with contextlib.redirect_stdout(io.StringIO()):
                res = self._compute(FIDUCIAL_OMBH2, 0.0, _raw_dNeff=True)
                self._linx.release_unused_memory()
            self._Neff_sm = res["Neff_BBN"]
            self.log.info(
                "LINX ready: Y_p=%.6f, D/H=%.4e, Neff=%.5f at ombh2=%.5f, dNeff=0.",
                res["Y_p"],
                res["DH"],
                res["Neff_BBN"],
                FIDUCIAL_OMBH2,
            )

        if self.dNeff_convention == "final":
            # Estimate the conversion slope from background solves at 0 and 1.
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, _, _, _, _, Neff_vec = self._bkg(self._jnp.asarray(1.0))
            self._dNeff_slope = float(Neff_vec[-1]) - self._Neff_sm
            self.log.info(
                "dNeff_convention='final': sampled dNeff is the post-e+e- value; "
                "Delt_Neff_init = dNeff / %.5f. (N_eff(SM) = %.5f.)",
                self._dNeff_slope,
                self._Neff_sm,
            )
        else:
            self.log.info(
                "dNeff_convention='init': the sampled dNeff is LINX's native "
                "pre-e+e--annihilation parameter, so the final Delta N_eff "
                "reaching the Boltzmann code is smaller by the entropy-transfer "
                "factor. Check your prior range is what you mean."
            )

    # --- parameter plumbing ----------------------------------------------

    def get_can_support_params(self):
        params = ["ombh2", "dNeff"]
        if self.sample_nuclear:
            params += ["tau_n_fac", *self._q_param_names]
        return params

    def get_can_provide_params(self):
        provided = ["YHe", "Y_p", "DH", "Neff_BBN", "N_ur"]
        if self.provide_nnu:
            provided.append("nnu")
        return provided

    # --- the actual computation -------------------------------------------

    def _delt_neff_init(self, dNeff):
        """Convert the sampled dNeff to LINX's native Delt_Neff_init."""
        if self.dNeff_convention == "final":
            return dNeff / self._dNeff_slope
        return dNeff

    def _compute(
        self,
        ombh2,
        dNeff,
        tau_n_fac=1.0,
        nuclear_rates_q=None,
        _raw_dNeff=False,
    ):
        """
        Run LINX at one point. Returns a dict of derived parameters.

        ``_raw_dNeff`` bypasses the convention conversion and is used during
        initialization, before the calibration slope exists.
        """
        jnp = self._jnp
        if not _raw_dNeff:
            dNeff = self._delt_neff_init(dNeff)

        if nuclear_rates_q is None:
            nuclear_rates_q = self._q0
        else:
            nuclear_rates_q = jnp.asarray(nuclear_rates_q, dtype=jnp.float64)

        # eta_fac = ombh2 / 0.02242. LINX's baryon mass assumes a fixed
        # Y_p_0 = 0.247 and T_CMB = 2.7255 K (const.py:48-59), which is its
        # documented convention. If TCMB is ever varied, this needs the
        # (2.7255/TCMB)**3 rescaling CAMB applies at camb/model.py:733.
        eta_fac = ombh2 / self._ombh2_ref

        (
            t_vec,
            a_vec,
            rho_g_vec,
            rho_nu_vec,
            rho_NP_vec,
            P_NP_vec,
            Neff_vec,
        ) = self._bkg(jnp.asarray(dNeff, dtype=jnp.float64))

        # t_vec/a_vec must be passed explicitly, otherwise AbundanceModel runs
        # two extra ODE solves to recompute them (abundances.py:215-219).
        sol = self._abd(
            rho_g_vec,
            rho_nu_vec,
            rho_NP_vec,
            P_NP_vec,
            t_vec=t_vec,
            a_vec=a_vec,
            eta_fac=jnp.asarray(eta_fac, dtype=jnp.float64),
            tau_n_fac=jnp.asarray(tau_n_fac, dtype=jnp.float64),
            nuclear_rates_q=nuclear_rates_q,
            rtol=self.rtol,
            atol=self.atol,
            sampling_nTOp=self.sampling_nTOp,
            max_steps=self.max_steps,
        )

        # Indices resolved by name in initialize(), not assumed positionally.
        Y_p = float(4.0 * sol[self._i_He4])          # nucleon fraction (CAMB's Y_p)
        DH = float(sol[self._i_d] / sol[self._i_p])  # n_D / n_H

        # The Boltzmann code needs the *final* N_eff, not 3.044 + dNeff. By this
        # point the local dNeff is Delt_Neff_init (T ~ 8.6 MeV, before neutrino
        # decoupling) whichever convention the sampler used, so adding it to
        # 3.044 would be wrong under either one.
        Neff_BBN = float(Neff_vec[-1])

        return {
            "YHe": float(self._ypBBN_to_yhe(Y_p)),
            "Y_p": Y_p,
            "DH": DH,
            "Neff_BBN": Neff_BBN,
        }

    def calculate(self, state, want_derived=True, **params_values_dict):
        ombh2 = params_values_dict["ombh2"]
        dNeff = params_values_dict.get("dNeff", 0.0)
        if self.sample_nuclear:
            tau_n_fac = params_values_dict["tau_n_fac"]
            nuclear_rates_q = [
                params_values_dict[name] for name in self._q_param_names
            ]
        else:
            tau_n_fac = 1.0
            nuclear_rates_q = None

        derived = self._compute(
            ombh2,
            dNeff,
            tau_n_fac=tau_n_fac,
            nuclear_rates_q=nuclear_rates_q,
        )

        if not all(np.isfinite(v) for v in derived.values()):
            self.log.debug(
                "LINX returned non-finite output at ombh2=%r, dNeff=%r; "
                "assigning zero likelihood.",
                ombh2,
                dNeff,
            )
            return False

        if self.provide_nnu:
            derived["nnu"] = derived["Neff_BBN"]
        derived["N_ur"] = derived["Neff_BBN"] - self.nur_shift

        state["derived"] = derived
        return True
