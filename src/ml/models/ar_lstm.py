from __future__ import annotations

import logging
import math
import random
from typing import Final
import polars as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from ml.models.lstm import QuantileLSTM, huber_pinball_loss, create_windows, EXCLUDE, CLASSES, CLASS_CODES, set_seed

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

class ARLSTMModel:
    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons
        
        # Only use seq_cols, no cross_cols because we can't easily auto-regress engineered features
        self.seq_cols = ["log_value"]
        self.cross_cols = []
        all_cols_to_scale = self.seq_cols
        
        mu_expr = [pl.col(c).mean().alias(c) for c in all_cols_to_scale]
        sd_expr = [pl.col(c).std().alias(c) for c in all_cols_to_scale]
        
        mu_df = train.select(mu_expr)
        sd_df = train.select(sd_expr)
        
        self.mu = {c: mu_df[c][0] for c in all_cols_to_scale}
        self.sd = {c: sd_df[c][0] for c in all_cols_to_scale}
        
        train_norm = self._preprocess(train)
        
        # AR model is ONLY trained for h=1
        h = 1
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
        LOGGER.info(f"Training ARLSTM on device: {device} for h=1")

        model = QuantileLSTM(n_seq_features=len(self.seq_cols), n_cross_features=len(self.cross_cols), hidden=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
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
        self.model = model

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
        model = self.model
        model.to(device)
        
        for split_name, df in eval_frames.items():
            df_norm = self._preprocess(df)
            
            for h in self.horizons:
                # We start with the original h=1 window
                seq_eval, cross_eval, c_eval, _, meta_eval = create_windows(df_norm, 1, self.seq_cols, self.cross_cols)
                
                if len(seq_eval) > 0:
                    seq_eval = seq_eval.to(device)
                    cross_eval = cross_eval.to(device)
                    c_eval = c_eval.to(device)
                    
                    with torch.no_grad():
                        current_seq = seq_eval.clone()
                        
                        # We must track the accumulated return relative to step 0
                        # Shape: (batch, 3). Initially 0.
                        total_ret = torch.zeros((seq_eval.shape[0], 3), device=device)
                        
                        # Autoregressive loop for h steps
                        for step in range(h):
                            preds_ret = model(current_seq, cross_eval, c_eval) # shape: (batch, 3)
                            total_ret = total_ret + preds_ret
                            
                            if step < h - 1:
                                # We need to update the sequence
                                # preds_ret is the UN-normalized return.
                                # The next normalized log_value = current normalized log_value + (preds_ret (median) / sd)
                                last_log_val = current_seq[:, -1, 0] # shape: (batch,)
                                median_ret = preds_ret[:, 1]
                                
                                sd_val = float(self.sd["log_value"])
                                if sd_val == 0 or math.isnan(sd_val):
                                    sd_val = 1.0
                                    
                                next_log_val = (last_log_val + (median_ret / sd_val)).unsqueeze(-1).unsqueeze(-1) # shape: (batch, 1, 1)
                                
                                # Shift sequence left and append new value
                                current_seq = torch.cat([current_seq[:, 1:, :], next_log_val], dim=1)
                                
                        preds = total_ret.cpu()
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
