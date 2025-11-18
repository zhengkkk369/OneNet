"""GLAFF-D3A fusion experiment for PatchTST backbone."""
from exp.exp_glaff_d3a_mixin import D3AFusionMixin
from exp.exp_patch import Exp_TS2VecSupervised as BaseExp


class Exp_TS2VecSupervised(D3AFusionMixin, BaseExp):
    def __init__(self, args):
        BaseExp.__init__(self, args)
        self._init_d3a(args)

    def test(self, setting, test=0):  # type: ignore[override]
        return self._d3a_test(setting, test)
