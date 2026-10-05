from __future__ import annotations

import logging
import math
import random
from typing import Final

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

EXCLUDE: Final[frozenset[str]] = frozenset(
    {"date", "target_value", "log_value", "y_step_h7", "y_step_h30", "y_step_h90",
     "y_mean_h7", "y_mean_h30", "y_mean_h90", "target_class"}
)

CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")
CLASS_CODES: Final[dict[str, int]] = {c: i for i, c in enumerate(CLASSES)}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuantileLSTM(nn.Module):
    def __init__(self, n_seq_features: int, n_cross_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.cls_emb = nn.Embedding(len(CLASSES), 2)
        
        # 1D-CNN (Local feature extraction)
        cnn_out_channels = 64
        self.conv1d = nn.Conv1d(in_channels=n_seq_features, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        
        # BiLSTM with LayerNorm
        self.lstm = nn.LSTM(cnn_out_channels, hidden, num_layers=2, batch_first=True, bidirectional=True)
        self.ln_pre = nn.LayerNorm(cnn_out_channels)
        self.ln_post = nn.LayerNorm(hidden * 2)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention Mechanism
        self.att_linear = nn.Linear(hidden * 2, hidden * 2)
        self.att_v = nn.Linear(hidden * 2, 1, bias=False)
        
        # Final head (hidden*2 + 2 for class emb + n_cross_features)
        self.heads = nn.Linear(hidden * 2 + 2 + n_cross_features, 3)

    def forward(self, x_seq: torch.Tensor, x_cross: torch.Tensor, cls_idx: torch.Tensor) -> torch.Tensor:
        # x_seq: (batch, seq_len, seq_features)
        x_cnn = x_seq.permute(0, 2, 1)  # (batch, features, seq_len)
        x_cnn = torch.relu(self.conv1d(x_cnn))
        x_cnn = self.dropout1(x_cnn)
        x_lstm_in = x_cnn.permute(0, 2, 1)  # (batch, seq_len, cnn_out)
        
        x_lstm_in = self.ln_pre(x_lstm_in)
        lstm_out, _ = self.lstm(x_lstm_in)  # (batch, seq_len, hidden * 2)
        lstm_out = self.ln_post(lstm_out)
        lstm_out = self.dropout2(lstm_out)
        
        # Attention over time steps
        u = torch.tanh(self.att_linear(lstm_out))
        att_scores = self.att_v(u)
        
        # Scale scores to prevent vanishing gradients in attention
        scale_factor = (self.lstm.hidden_size * 2) ** 0.5
        att_scores = att_scores / scale_factor
        
        att_weights = torch.softmax(att_scores, dim=1)
        
        context = torch.sum(lstm_out * att_weights, dim=1)  # (batch, hidden * 2)
        
        emb = self.cls_emb(cls_idx)  # (batch, 2)
        summary_emb = torch.cat([context, emb, x_cross], dim=-1)
        return self.heads(summary_emb)


def huber_pinball_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.05) -> torch.Tensor:
    target = target.unsqueeze(1)
    err = target - pred
    q = torch.tensor([0.1, 0.5, 0.9], device=pred.device).unsqueeze(0)
    
    abs_err = torch.abs(err)
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    huber = 0.5 * (quadratic ** 2) + delta * linear
    huber_scaled = huber / delta
    
    weights = torch.where(err >= 0, q, 1.0 - q)
    loss = weights * huber_scaled
    return loss.sum(dim=-1).mean()


def create_windows(df: pl.DataFrame, h: int, seq_cols: list[str], cross_cols: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[object, str]]]:
    # seq_cols are features fed into the sequence model (e.g., log_value)
    # cross_cols are features fed into the final MLP (e.g., rolling averages)
    
    # Reconstruct continuous calendar
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_grid = pl.DataFrame({"date": pl.date_range(min_date, max_date, "1d", eager=True)})
    
    seq_list, cross_list, y_list, cls_list, meta_list = [], [], [], [], []
    
    for cls_name, grp in df.group_by("target_class"):
        class_name = cls_name[0] if isinstance(cls_name, tuple) else cls_name
        grp = date_grid.join(grp, on="date", how="left").fill_null(strategy="forward").sort("date")
        
        valid_dates = set(df.filter(pl.col("target_class") == class_name)["date"].to_list())
        
        seq_arr = grp.select(seq_cols).to_torch()
        
        if cross_cols:
            cross_arr = grp.select(cross_cols).to_torch()
        else:
            cross_arr = torch.empty((grp.height, 0))
        
        if f"y_step_h{h}" in grp.columns:
            y_arr = (grp[f"y_step_h{h}"] - grp["log_value"]).to_torch()
        else:
            y_arr = torch.full((grp.height,), float('nan'))
        
        dates_arr = grp["date"].to_list()
        
        for i in range(120, grp.height + 1):
            if dates_arr[i - 1] not in valid_dates:
                continue
                
            window_seq = seq_arr[i - 120 : i]
            cross_feat = cross_arr[i - 1]
            target = y_arr[i - 1]
            
            seq_list.append(window_seq)
            cross_list.append(cross_feat)
            y_list.append(target if not math.isnan(target.item()) else float('nan'))
            cls_list.append(CLASS_CODES[str(class_name)])
            meta_list.append((dates_arr[i - 1], str(class_name)))

    if not seq_list:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), []
        
    seq_tensor = torch.stack(seq_list).to(torch.float32)
    cross_tensor = torch.stack(cross_list).to(torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)
    cls_tensor = torch.tensor(cls_list, dtype=torch.long)
    return seq_tensor, cross_tensor, cls_tensor, y_tensor, meta_list


class LSTMModel:
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons
        self.models = {}
        
        all_feature_cols = [c for c in train.columns if c not in EXCLUDE]
        
        # Split features into seq (raw) and cross (engineered)
        self.seq_cols = ["log_value"]
        self.cross_cols = [c for c in all_feature_cols if c not in self.seq_cols]
        all_cols_to_scale = self.seq_cols + self.cross_cols
        
        mu_expr = [pl.col(c).mean().alias(c) for c in all_cols_to_scale]
        sd_expr = [pl.col(c).std().alias(c) for c in all_cols_to_scale]
        
        mu_df = train.select(mu_expr)
        sd_df = train.select(sd_expr)
        
        self.mu = {c: mu_df[c][0] for c in all_cols_to_scale}
        self.sd = {c: sd_df[c][0] for c in all_cols_to_scale}
        
        train_norm = self._preprocess(train)
        
        for h in horizons:
            set_seed(42)
            seq_all, cross_all, c_all, y_all, meta_all = create_windows(train_norm, h, self.seq_cols, self.cross_cols)
            
            if len(seq_all) == 0:
                raise ValueError("No valid training windows found.")
                
            dates = [m[0] for m in meta_all]
            sort_idx = sorted(range(len(dates)), key=lambda k: dates[k])
            sort_idx = torch.tensor(sort_idx, dtype=torch.long)
            
            seq_all = seq_all[sort_idx]
            cross_all = cross_all[sort_idx]
            c_all = c_all[sort_idx]
            y_all = y_all[sort_idx]
            
            valid_mask = ~torch.isnan(y_all)
            seq_all = seq_all[valid_mask]
            cross_all = cross_all[valid_mask]
            c_all = c_all[valid_mask]
            y_all = y_all[valid_mask]
            
            core_n = max(int(len(seq_all) * 0.85), len(seq_all) - 60)
            
            seq_tr = seq_all[:core_n]
            cross_tr = cross_all[:core_n]
            c_tr = c_all[:core_n]
            y_tr = y_all[:core_n]
            
            seq_es = seq_all[core_n:]
            cross_es = cross_all[core_n:]
            c_es = c_all[core_n:]
            y_es = y_all[core_n:]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            LOGGER.info(f"Training on device: {device} for h={h}")

            model = QuantileLSTM(n_seq_features=len(self.seq_cols), n_cross_features=len(self.cross_cols), hidden=32).to(device)
            # Lower learning rate for longer horizons
            lr = 1e-4 if h >= 30 else 1e-3
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            tr_ds = TensorDataset(seq_tr, cross_tr, c_tr, y_tr)
            tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
            
            if len(seq_es) > 0:
                seq_es, cross_es, c_es, y_es = seq_es.to(device), cross_es.to(device), c_es.to(device), y_es.to(device)
                
            best_loss = float("inf")
            patience = 8
            patience_counter = 0
            best_state = None
            
            for epoch in range(100):
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
                        pred_es = model(seq_es, cross_es, c_es)
                        val_loss = huber_pinball_loss(pred_es, y_es).item()
                    else:
                        val_loss = 0.0
                        
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
                        
            if best_state is not None:
                model.load_state_dict(best_state)
            
            model.eval()
            self.models[h] = model

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

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        out: dict[str, dict[int, pl.DataFrame]] = {name: {} for name in eval_frames}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for h in self.horizons:
            model = self.models[h]
            model.to(device)
            
            for split_name, df in eval_frames.items():
                df_norm = self._preprocess(df)
                seq_eval, cross_eval, c_eval, _, meta_eval = create_windows(df_norm, h, self.seq_cols, self.cross_cols)
                
                if len(seq_eval) > 0:
                    seq_eval, cross_eval, c_eval = seq_eval.to(device), cross_eval.to(device), c_eval.to(device)
                    with torch.no_grad():
                        preds = model(seq_eval, cross_eval, c_eval).cpu()
                        preds_0 = preds[:, 0].tolist()
                        preds_1 = preds[:, 1].tolist()
                        preds_2 = preds[:, 2].tolist()
                else:
                    preds_0 = []
                    preds_1 = []
                    preds_2 = []
                    
                dates_list = [m[0] for m in meta_eval]
                classes_list = [m[1] for m in meta_eval]
                
                pred_df = pl.DataFrame({
                    "date": dates_list,
                    "target_class": classes_list,
                    "p_0.1_ret": preds_0,
                    "p_0.5_ret": preds_1,
                    "p_0.9_ret": preds_2,
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

