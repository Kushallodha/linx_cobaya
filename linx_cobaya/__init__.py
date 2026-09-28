"""Cobaya components wrapping LINX for BBN predictions."""

from .bbn_likelihood import BBNAbundances
from .linx_bbn import LinxBBN

__all__ = ["LinxBBN", "BBNAbundances"]
