"""Informer experiment wrapper.

This module currently reuses the FEDformer TS2Vec-supervised pipeline to satisfy
GLAFF/TS2Vec dispatcher imports. The architecture-level differences remain in
models/Informer.py, which is invoked via model selection in the shared
experiment class.
"""
from exp.exp_fedformer import Exp_TS2VecSupervised  # re-export

__all__ = ["Exp_TS2VecSupervised"]
