"""GLAFF fusion experiment dispatcher.

This module keeps the experiment entry point name stable (``Exp_TS2VecSupervised``)
while routing to the appropriate backbone-specific experiment class. Because the
GLAFF plugin is already integrated inside model definitions, no additional
wrapper model is required here; we simply select the desired backbone experiment
based on ``args.glaff_backbone``.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple, Type

import torch

from exp.exp_basic import Exp_Basic
from exp.exp_cross_former import Exp_TS2VecSupervised as ExpCrossFormer
from exp.exp_dlinear import Exp_TS2VecSupervised as ExpDLinear
from exp.exp_fedformer import Exp_TS2VecSupervised as ExpFedformer
from exp.exp_informer import Exp_TS2VecSupervised as ExpInformer
from exp.exp_patch import Exp_TS2VecSupervised as ExpPatch
from exp.exp_ts2vec import Exp_TS2VecSupervised as ExpTS2Vec
from utils.metrics import cumavg, metric

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
    """Dispatch to the proper experiment class per backbone for GLAFF fusion.

    This wrapper keeps the drift-aware D3A-style evaluation loop that was
    previously present in this file while delegating model definitions to the
    backbone-specific experiment classes (which already embed the GLAFF plugin
    when ``args.flag == 'Plugin'``).
    """

    def __init__(self, args):
        backbone = _resolve_backbone(getattr(args, 'glaff_backbone', 'ts2vec'))
        if backbone not in _BACKBONE_DISPATCH:
            raise ValueError(
                f"Unsupported glaff_backbone '{backbone}'. Available: {sorted(_BACKBONE_DISPATCH.keys())}"
            )
        print(f"[GLAFF Fusion] Using backbone: {backbone}")
        self._exp = _BACKBONE_DISPATCH[backbone](args)

        # D3A-style drift detection parameters
        self.virtual_threshold = args.energy_threshold
        self.virtual_min_samples = args.virtual_min_samples
        self.residual_base_sigma = args.residual_base_sigma
        self.residual_mu_thresh = args.residual_mu_thresh
        self.residual_sigma_thresh = args.residual_sigma_thresh
        self.residual_window = args.residual_window
        self.buffer_limit = args.glaff_buffer_size
        self.ft_lr = args.glaff_ft_lr
        self.ft_epochs = args.glaff_ft_epochs

        self.feature_buffer: Deque[torch.Tensor] = deque(maxlen=self.buffer_limit)
        self.sample_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.residuals: Deque[torch.Tensor] = deque(maxlen=self.residual_window)
        self._latest_dataset = None

    def __getattr__(self, name):
        return getattr(self._exp, name)

    def __repr__(self):
        return f"Exp_TS2VecSupervised(dispatch={self._exp!r})"

    # ------------------------- Drift detection helpers ----------------------
    def _energy_distance(self, ref: torch.Tensor, new: torch.Tensor) -> float:
        dist_ref_new = torch.cdist(new, ref, p=2).mean()
        dist_new_new = torch.cdist(new, new, p=2).mean()
        dist_ref_ref = torch.cdist(ref, ref, p=2).mean()
        return float(2 * dist_ref_new - dist_new_new - dist_ref_ref)

    def _detect_virtual_drift(self) -> Tuple[bool, float]:
        if len(self.feature_buffer) < self.virtual_min_samples:
            return False, 0.0
        mid = len(self.feature_buffer) // 2
        ref = torch.cat(list(self.feature_buffer)[:mid], dim=0)
        new = torch.cat(list(self.feature_buffer)[mid:], dim=0)
        score = self._energy_distance(ref, new)
        return score > self.virtual_threshold, score

    def _detect_real_drift(self) -> Tuple[bool, float, float]:
        if len(self.residuals) < self.residual_window:
            return False, 0.0, 0.0
        residual_tensor = torch.cat(list(self.residuals), dim=0)
        mu = residual_tensor.mean().item()
        sigma = residual_tensor.std(unbiased=False).item()
        mu_flag = abs(mu) > self.residual_mu_thresh * self.residual_base_sigma
        sigma_flag = sigma > self.residual_sigma_thresh * self.residual_base_sigma
        return (mu_flag or sigma_flag), mu, sigma

    def _fine_tune_local(self):
        if not self.sample_buffer:
            return
        model = getattr(self._exp, 'model', None)
        if model is None:
            return
        criterion = getattr(self._exp, '_select_criterion', lambda: torch.nn.MSELoss())()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.ft_lr)
        model.train()
        dataset = self._latest_dataset
        for _ in range(self.ft_epochs):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.sample_buffer:
                pred, true = self._exp._process_one_batch(
                    dataset, batch_x.clone(), batch_y.clone(), batch_x_mark.clone(), batch_y_mark.clone(),
                )
                loss = criterion(pred, true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()

    # ----------------------------- Evaluation -------------------------------
    def test(self, setting, test=0):  # type: ignore[override]
        test_data, test_loader = self._exp._get_data(flag='test')
        self._latest_dataset = test_data
        self._exp.model.eval()

        preds, trues = [], []
        maes, mses, rmses, mapes, mspes = [], [], [], [], []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            # virtual drift queue uses flattened raw inputs
            self.feature_buffer.append(batch_x.view(batch_x.size(0), -1).detach())
            virtual_drift, energy_score = self._detect_virtual_drift()

            pred, true = self._exp._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test'
            )
            residual = (true - pred).detach()
            self.residuals.append(residual.view(residual.size(0), -1))

            # maintain sample buffer for possible fine-tuning
            self.sample_buffer.append((batch_x, batch_y, batch_x_mark, batch_y_mark))
            if len(self.sample_buffer) > self.buffer_limit:
                self.sample_buffer.pop(0)

            real_drift, mu, sigma = self._detect_real_drift()
            if real_drift:
                print(f"[GLAFF D3A] Real drift detected (mu={mu:.4f}, sigma={sigma:.4f}); fine-tuning local branch.")
                self._fine_tune_local()
                self.residuals.clear()
            elif virtual_drift:
                print(f"[GLAFF D3A] Virtual drift warning (energy={energy_score:.4f}).")

            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds_np = torch.cat(preds, dim=0).numpy()
        trues_np = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds_np.shape, trues_np.shape)
        MAE, MSE, RMSE, MAPE, MSPE = (
            cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        )
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]
        print('mse:{}, mae:{}'.format(mse, mae))
        return [mae, mse, rmse, mape, mspe], MAE, MSE, preds_np, trues_np
