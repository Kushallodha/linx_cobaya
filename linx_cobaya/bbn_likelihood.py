"""
Gaussian likelihood on the primordial abundances predicted by LINX.

Requires a component providing ``DH`` and ``Y_p`` -- i.e. ``LinxBBN``.
"""

import numpy as np

from cobaya.likelihood import Likelihood


class BBNAbundances(Likelihood):
    """
    Independent Gaussian constraints on D/H and Y_p.

    Defaults follow arXiv:2609.12065: Cooke et al. 2018 (1710.11129) for
    deuterium and Aver et al. 2026 for helium; override from YAML.
    """

    type = "BBN"

    use_DH: bool = True
    use_Yp: bool = True

    DH_mean: float = 2.527e-5
    DH_std: float = 0.030e-5
    DH_theory_std: float = 0.0

    Yp_mean: float = 0.2458
    Yp_std: float = 0.0013
    Yp_theory_std: float = 0.0

    def initialize(self):
        if not (self.use_DH or self.use_Yp):
            raise ValueError(
                "BBNAbundances: both use_DH and use_Yp are False, "
                "so the likelihood would be empty."
            )
        # Measurement and theory errors added in quadrature. Theory errors
        # default to 0 (2609.12065 measurement-only form). Set them when
        # nuclear rates are not being marginalised over and you want the BBN
        # theory error folded in.
        self._DH_std = float(np.hypot(self.DH_std, self.DH_theory_std))
        self._Yp_std = float(np.hypot(self.Yp_std, self.Yp_theory_std))

    def get_requirements(self):
        reqs = {}
        if self.use_DH:
            reqs["DH"] = None
        if self.use_Yp:
            reqs["Y_p"] = None
        return reqs

    def logp(self, **params_values):
        chi2 = 0.0
        if self.use_DH:
            chi2 += ((params_values["DH"] - self.DH_mean) / self._DH_std) ** 2
        if self.use_Yp:
            chi2 += ((params_values["Y_p"] - self.Yp_mean) / self._Yp_std) ** 2
        return -0.5 * chi2
