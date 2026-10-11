from __future__ import annotations

import logging
import math
from typing import Final

import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.models.lstm import (
    CLASSES,
    QuantileLSTM,
    add_target_returns,
    create_windows,
    huber_pinball_loss,
    set_seed,
    standardized_halfwidths,
    target_stats,
)

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

#: AR-LSTM is an h=1 model that is rolled forward; one seed keeps it cheap.
DEFAULT_AR_SEEDS: Final[tuple[int, ...]] = (42,)


class ARLSTMModel:
    """Autoregressive LSTM: train on the standardised h=1 return, roll forward.

    Mirrors :class:`ml.models.lstm.LSTMModel`'s monotone head, exact pinball
    loss and per-class target standardisation, but only trains h=1 and then
    feeds its own median prediction back in for h>1.
    """

    def __init__(self, max_epochs: int = 100, patience: int = 8, hidden: int = 32,
                 seeds: tuple[int, ...] = DEFAULT_AR_SEEDS) -> None:
        self.max_epochs = max_epochs
        self.patience = patience
        self.hidden = hidden
        self.seeds = seeds

    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons

        # Only use seq_cols, no cross_cols because we can't easily auto-regress engineered features
        self.seq_cols = ["log_value"]
        self.cross_cols = []
        all_cols_to_scale = self.seq_cols

        mu_df = train.select([pl.col(c).mean().alias(c) for c in all_cols_to_scale])
        sd_df = train.select([pl.col(c).std().alias(c) for c in all_cols_to_scale])

        self.mu = {c: mu_df[c][0] for c in all_cols_to_scale}
        self.sd = {c: sd_df[c][0] for c in all_cols_to_scale}

        train_aug = add_target_returns(train, [1])
        train_norm = self._preprocess(train_aug)

        # AR model is ONLY trained for h=1
        h = 1
        self.target_stats = {h: target_stats(train, h)}
        low_w, high_w = standardized_halfwidths(train, h)
        seq_all, cross_all, c_all, y_all, meta_all = create_windows(
            train_norm, h, self.seq_cols, self.cross_cols, target_col="target_ret_h1"
        )

        if len(seq_all) == 0:
            raise ValueError("No valid training windows found.")

        dates = [m[0] for m in meta_all]
        sort_idx = sorted(range(len(dates)), key=lambda k: dates[k])
        sort_idx = torch.tensor(sort_idx, dtype=torch.long)

        seq_all = seq_all[sort_idx]
        cross_all = cross_all[sort_idx]
        c_all = c_all[sort_idx]
        y_all = y_all[sort_idx]
        meta_all = [meta_all[i] for i in sort_idx.tolist()]

        mu = torch.tensor([self.target_stats[h][m[1]][0] for m in meta_all])
        sd = torch.tensor([self.target_stats[h][m[1]][1] for m in meta_all])
        y_all = (y_all - mu) / sd

        valid_mask = ~torch.isnan(y_all)
        seq_all = seq_all[valid_mask]
        cross_all = cross_all[valid_mask]
        c_all = c_all[valid_mask]
        y_all = y_all[valid_mask]

        core_n = max(int(len(seq_all) * 0.85), len(seq_all) - 60)

        seq_tr, cross_tr, c_tr, y_tr = seq_all[:core_n], cross_all[:core_n], c_all[:core_n], y_all[:core_n]
        seq_es, cross_es, c_es, y_es = seq_all[core_n:], cross_all[core_n:], c_all[core_n:], y_all[core_n:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Training ARLSTM on device: %s for h=1", device)

        self.models: list[tuple[int, QuantileLSTM]] = []
        for seed in self.seeds:
            set_seed(seed)
            model = QuantileLSTM(n_seq_features=len(self.seq_cols), n_cross_features=0, hidden=self.hidden)
            model.init_quantile_biases(low_w, high_w)
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            tr_ds = TensorDataset(seq_tr, cross_tr, c_tr, y_tr)
            tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)

            if len(seq_es) > 0:
                seq_es_d = seq_es.to(device)
                cross_es_d = cross_es.to(device)
                c_es_d = c_es.to(device)
                y_es_d = y_es.to(device)

            best_loss = float("inf")
            patience_counter = 0
            best_state = None

            for epoch in range(self.max_epochs):
                model.train()
                for b_seq, b_cross, bc, by in tr_dl:
                    b_seq, b_cross, bc, by = b_seq.to(device), b_cross.to(device), bc.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(b_seq, b_cross, bc)
                    loss = huber_pinball_loss(pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    if len(seq_es) > 0:
                        pred_es = model(seq_es_d, cross_es_d, c_es_d)
                        val_loss = huber_pinball_loss(pred_es, y_es_d).item()
                    else:
                        val_loss = 0.0

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self.models.append((seed, model))

    def _preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        all_cols = self.seq_cols + self.cross_cols
        for c in all_cols:
            m, s = self.mu[c], self.sd[c]
            if s is None or s == 0 or math.isnan(s):
                s = 1.0
            if m is None or math.isnan(m):
                m = 0.0
            exprs.append(
                ((pl.col(c) - m) / s).fill_nan(0.0).fill_null(0.0).alias(c)
            )
        return df.with_columns(exprs)

    def _rollout(
        self, model: QuantileLSTM, seq_eval: torch.Tensor, cls_eval: torch.Tensor,
        h: int, device: torch.device,
    ) -> torch.Tensor:
        """Roll the h=1 model forward ``h`` steps, accumulating raw returns."""
        stats = self.target_stats[1]
        sd1 = torch.tensor([stats[c][1] for c in CLASSES], device=device)[cls_eval]
        mu1 = torch.tensor([stats[c][0] for c in CLASSES], device=device)[cls_eval]
        empty_cross = torch.empty((seq_eval.shape[0], 0), device=device)
        sd_log = float(self.sd["log_value"])
        if sd_log == 0 or math.isnan(sd_log):
            sd_log = 1.0

        with torch.no_grad():
            current_seq = seq_eval.clone()
            total = torch.zeros((seq_eval.shape[0], 3), device=device)
            for step in range(h):
                preds_z = model(current_seq, empty_cross, cls_eval)
                # Convert standardised quantiles to raw h=1 returns per class.
                raw = preds_z * sd1.unsqueeze(1) + mu1.unsqueeze(1)
                total = total + raw
                if step < h - 1:
                    last_z = current_seq[:, -1, 0]
                    median_raw = raw[:, 1]
                    next_z = (last_z + (median_raw / sd_log)).unsqueeze(-1).unsqueeze(-1)
                    current_seq = torch.cat([current_seq[:, 1:, :], next_z], dim=1)
        return total.cpu()

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for split_name, df in eval_frames.items():
            df_norm = self._preprocess(add_target_returns(df, [1]))

            for h in self.horizons:
                seq_eval, cross_eval, c_eval, _, meta_eval = create_windows(
                    df_norm, 1, self.seq_cols, self.cross_cols, target_col="target_ret_h1"
                )

                if len(seq_eval) > 0:
                    seq_eval = seq_eval.to(device)
                    c_eval = c_eval.to(device)
                    seed_preds = []
                    for _, model in self.models:
                        model.to(device)
                        seed_preds.append(self._rollout(model, seq_eval, c_eval, h, device))
                    preds = torch.stack(seed_preds, dim=0).mean(dim=0)
                    p10r = preds[:, 0].tolist()
                    p50r = preds[:, 1].tolist()
                    p90r = preds[:, 2].tolist()
                else:
                    p10r = p50r = p90r = []

                pred_df = pl.DataFrame({
                    "date": [m[0] for m in meta_eval],
                    "target_class": [m[1] for m in meta_eval],
                    "p_0.1_ret": p10r,
                    "p_0.5_ret": p50r,
                    "p_0.9_ret": p90r,
                })

                df_join = df.select(["date", "target_class", "log_value"])
                res = df_join.join(pred_df, on=["date", "target_class"], how="left")

                res = res.select([
                    pl.col("date"),
                    pl.col("target_class"),
                    pl.lit(h).cast(pl.Int64).alias("h"),
                    (pl.col("log_value") + pl.col("p_0.1_ret")).alias("p_0.1"),
                    (pl.col("log_value") + pl.col("p_0.5_ret")).alias("p_0.5"),
                    (pl.col("log_value") + pl.col("p_0.9_ret")).alias("p_0.9"),
                ])

                res = res.drop_nulls(["p_0.5"])
                out[split_name][h] = res

        return out
