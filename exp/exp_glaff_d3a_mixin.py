"""Shared D3A-style drift handling for GLAFF fusion experiments."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple

import torch

from utils.metrics import cumavg, metric


class D3AFusionMixin:
    """Mixin that adds concept-drift-aware testing to backbone experiments."""

    def _init_d3a(self, args):
        # thresholds and buffer sizing
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
        model = getattr(self, 'model', None)
        if model is None:
            return
        criterion = getattr(self, '_select_criterion', lambda: torch.nn.MSELoss())()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.ft_lr)
        model.train()
        dataset = self._latest_dataset
        for _ in range(self.ft_epochs):
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.sample_buffer:
                pred, true = self._process_one_batch(
                    dataset, batch_x.clone(), batch_y.clone(), batch_x_mark.clone(), batch_y_mark.clone(),
                )
                loss = criterion(pred, true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        model.eval()

    # ----------------------------- Evaluation -------------------------------
    def _d3a_test(self, setting, test=0):  # type: ignore[override]
        test_data, test_loader = self._get_data(flag='test')
        self._latest_dataset = test_data
        self.model.eval()

        preds, trues = [], []
        maes, mses, rmses, mapes, mspes = [], [], [], [], []

        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            self.feature_buffer.append(batch_x.view(batch_x.size(0), -1).detach())
            virtual_drift, energy_score = self._detect_virtual_drift()

            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test'
            )
            residual = (true - pred).detach()
            self.residuals.append(residual.view(residual.size(0), -1))

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
