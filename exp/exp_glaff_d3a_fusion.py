"""GLAFF fusion experiment dispatcher.

This module keeps the experiment entry point name stable (``Exp_TS2VecSupervised``)
while routing to the appropriate backbone-specific experiment class. Because the
GLAFF plugin is already integrated inside model definitions, no additional
wrapper model is required here; we simply select the desired backbone experiment
based on ``args.glaff_backbone``.
"""
from __future__ import annotations

from typing import Dict, Type

from exp.exp_basic import Exp_Basic
from exp.exp_cross_former import Exp_TS2VecSupervised as ExpCrossFormer
from exp.exp_dlinear import Exp_TS2VecSupervised as ExpDLinear
from exp.exp_fedformer import Exp_TS2VecSupervised as ExpFedformer
from exp.exp_informer import Exp_TS2VecSupervised as ExpInformer
from exp.exp_patch import Exp_TS2VecSupervised as ExpPatch
from exp.exp_ts2vec import Exp_TS2VecSupervised as ExpTS2Vec

# FEDformer/Autoformer/Informer share the same training-testing pipeline in the
# existing experiment implementations, so they can point to the same class.
_BACKBONE_DISPATCH: Dict[str, Type[Exp_Basic]] = {
    'ts2vec': ExpTS2Vec,
    'dlinear': ExpDLinear,
    'patchtst': ExpPatch,
    'fedformer': ExpFedformer,
    'autoformer': ExpFedformer,
    'informer': ExpInformer,
    'crossformer': ExpCrossFormer,
}


def _resolve_backbone(name: str) -> str:
    """Normalize backbone names for dispatching."""

    return str(name).lower()


class Exp_TS2VecSupervised:
    """Dispatch to the proper experiment class per backbone for GLAFF fusion."""

    def __init__(self, args):
        backbone = _resolve_backbone(getattr(args, 'glaff_backbone', 'ts2vec'))
        if backbone not in _BACKBONE_DISPATCH:
            raise ValueError(
                f"Unsupported glaff_backbone '{backbone}'. Available: {sorted(_BACKBONE_DISPATCH.keys())}"
            )
        # Echo the resolved backbone to make experiment selection transparent in logs
        print(f"[GLAFF Fusion] Using backbone: {backbone}")
        self._exp = _BACKBONE_DISPATCH[backbone](args)

    def __getattr__(self, name):
        return getattr(self._exp, name)

    def __repr__(self):
        return f"Exp_TS2VecSupervised(dispatch={self._exp!r})"
