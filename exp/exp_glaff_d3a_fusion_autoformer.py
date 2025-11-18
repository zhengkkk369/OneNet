"""Backbone-specific entry point for GLAFF-D3A fusion."""
from exp.exp_glaff_d3a_fusion import Exp_TS2VecSupervised as _Base


class Exp_TS2VecSupervised(_Base):
    def __init__(self, args):
        args.glaff_backbone = __name__.split('_')[-1]
        super().__init__(args)
