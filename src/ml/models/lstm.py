from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
from pathlib import Path
from typing import Final, Self

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

HORIZONS: Final[tuple[int, ...]] = (1, 7, 30, 90)

#: Quantile levels produced by the head, in ascending order.
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)

#: Repository root (src/ml/models/lstm.py -> parents[3]).
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
CHECKPOINT_DIR: Final[Path] = REPO_ROOT / "src" / "data" / "checkpoints"

# Every target-derived column must be excluded from the feature matrix. In
# particular ``y_step_h1`` used to leak the h=1 target straight into the model.
_TARGET_COLS: Final[frozenset[str]] = frozenset(
    {f"y_step_h{h}" for h in HORIZONS}
    | {f"y_mean_h{h}" for h in HORIZONS}
    | {f"target_ret_h{h}" for h in HORIZONS}
)

EXCLUDE: Final[frozenset[str]] = frozenset(
    {"date", "target_value", "log_value", "target_class"}
) | _TARGET_COLS

CLASSES: Final[tuple[str, ...]] = ("Capesize", "Panamax", "Supramax", "Handysize")
CLASS_CODES: Final[dict[str, int]] = {c: i for i, c in enumerate(CLASSES)}

#: Default number of independently seeded LSTM ensembles.
DEFAULT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

#: Seconds without a per-epoch heartbeat before the watchdog aborts the run.
#: A hard GPU stall leaves no other way to reclaim the process; per-epoch
#: checkpoints mean the run can be resumed.
STALL_TIMEOUT_S: Final[float] = float(os.environ.get("LSTM_STALL_TIMEOUT_S", "600"))


def set_seed(seed: int = 42) -> None:
    """Seed every RNG used by training and enable deterministic kernels.

    ``np.random.seed`` was previously missing, so the (small) numpy-dependent
    paths and any DataLoader worker seeding were not reproducible.
    ``CUBLAS_WORKSPACE_CONFIG`` is required for deterministic cuBLAS on CUDA
    >= 10.2; it is set here (before the first cuBLAS handle is created).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _inv_softplus(y: float) -> float:
    """Stable inverse of ``softplus``; ``softplus(_inv_softplus(y)) == y``."""
    y = max(float(y), 1e-6)
    return math.log(math.expm1(y))


class QuantileLSTM(nn.Module):
    """CNN + BiLSTM + attention backbone with a monotone quantile head.

    The head emits three raw values ``(o1, o2, o3)`` and constructs the
    quantiles so that ordering holds by construction::

        p50 = o1
        p10 = p50 - softplus(o2)
        p90 = p50 + softplus(o3)

    ``o2``/``o3`` are initialised (through :meth:`init_quantile_biases`) so the
    untrained head starts at the train-set empirical half-widths for the
    horizon, rather than at ~0 widths that the old unconstrained head collapsed
    toward.
    """

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

        # Final head: (p50, log-width-low, log-width-high)
        self.heads = nn.Linear(hidden * 2 + 2 + n_cross_features, 3)

    def init_quantile_biases(self, low_halfwidth: float, high_halfwidth: float) -> None:
        """Set head biases so initial widths equal the given half-widths.

        ``low_halfwidth`` is ``(p50 - p10)`` and ``high_halfwidth`` is
        ``(p90 - p50)`` in the standardised target space for this horizon.
        """
        with torch.no_grad():
            self.heads.bias[0] = 0.0
            self.heads.bias[1] = _inv_softplus(low_halfwidth)
            self.heads.bias[2] = _inv_softplus(high_halfwidth)

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
        raw = self.heads(summary_emb)

        p50 = raw[:, 0]
        p10 = p50 - F.softplus(raw[:, 1])
        p90 = p50 + F.softplus(raw[:, 2])
        return torch.stack([p10, p50, p90], dim=1)


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: tuple[float, ...] = QUANTILES,
) -> torch.Tensor:
    """Exact pinball (quantile) loss, averaged over rows and quantile levels.

    ``pred`` is ``(batch, len(quantiles))`` in ascending-quantile order and
    ``target`` is ``(batch,)``.
    """
    target = target.unsqueeze(1)
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).unsqueeze(0)
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()


def huber_pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float | None = None,
    quantiles: tuple[float, ...] = QUANTILES,
) -> torch.Tensor:
    """Huber-smoothed pinball loss.

    ``delta=None`` (the default and the training default) returns the exact
    pinball loss. A finite ``delta`` smooths it, but the caller must keep
    ``delta`` small relative to the target scale: the old default
    ``delta=0.05`` was ~70% of the h=1 target std (0.069), which turned the
    loss into an expectile loss and shrank the outer quantiles toward P50.
    """
    if delta is None:
        return pinball_loss(pred, target, quantiles)

    target = target.unsqueeze(1)
    err = target - pred
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype).unsqueeze(0)

    abs_err = torch.abs(err)
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    huber = 0.5 * (quadratic ** 2) + delta * linear
    huber_scaled = huber / delta

    weights = torch.where(err >= 0, q, 1.0 - q)
    loss = weights * huber_scaled
    return loss.sum(dim=-1).mean()


def add_target_returns(df: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    """Append raw-scale h-step log-return targets ``y_step_h{h} - log_value``.

    The LSTM is trained on returns, but ``_preprocess`` standardises
    ``log_value`` (a sequence feature). Computing the target inside
    ``create_windows`` from the *normalised* frame mixed raw levels with
    standardised levels and silently turned the h>=7 target into the log level
    itself. Materialising the return on the raw frame before any scaling keeps
    prediction and target on the same scale.
    """
    exprs = [
        (pl.col(f"y_step_h{h}") - pl.col("log_value")).alias(f"target_ret_h{h}")
        for h in horizons
        if f"y_step_h{h}" in df.columns
    ]
    return df.with_columns(exprs) if exprs else df


def target_stats(train: pl.DataFrame, h: int) -> dict[str, tuple[float, float]]:
    """Per-class ``(mean, std)`` of the h-step log return on train rows only.

    Standardising per class *and* horizon keeps the head's output scale
    comparable across classes; the per-class statistics are stored and inverted
    at prediction time so the model never sees raw level-scale variation.
    """
    out: dict[str, tuple[float, float]] = {}
    for cls in CLASSES:
        e = (
            train.filter(pl.col("target_class") == cls)
            .select(pl.col(f"y_step_h{h}") - pl.col("log_value"))
            .to_series()
            .drop_nulls()
        )
        if e.len() > 1:
            mu = float(e.mean())
            sd = float(e.std())
        else:
            mu, sd = 0.0, 1.0
        if not math.isfinite(sd) or sd <= 1e-9:
            sd = 1.0
        out[cls] = (mu, sd)
    return out


def standardized_halfwidths(train: pl.DataFrame, h: int) -> tuple[float, float]:
    """Pooled empirical ``(q50-q10, q90-q50)`` of the standardised h-step return.

    Used to initialise the head's width biases so the untrained model starts
    near the empirical interval width instead of a degenerate point.
    """
    stats = target_stats(train, h)
    e = (
        train.select(
            "target_class",
            (pl.col(f"y_step_h{h}") - pl.col("log_value")).alias("e"),
        )
        .drop_nulls("e")
        .with_columns(
            pl.col("target_class").replace_strict(
                {c: stats[c][0] for c in CLASSES}
            ).alias("_mu"),
            pl.col("target_class").replace_strict(
                {c: stats[c][1] for c in CLASSES}
            ).alias("_sd"),
        )
        .with_columns(((pl.col("e") - pl.col("_mu")) / pl.col("_sd")).alias("z"))
    )
    if e.height < 3:
        return 0.1, 0.1
    low = float(e["z"].quantile(0.5)) - float(e["z"].quantile(0.1))
    high = float(e["z"].quantile(0.9)) - float(e["z"].quantile(0.5))
    return max(low, 1e-3), max(high, 1e-3)


def create_windows(df: pl.DataFrame, h: int, seq_cols: list[str], cross_cols: list[str], target_col: str | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[object, str]]]:
    # seq_cols are features fed into the sequence model (e.g., log_value)
    # cross_cols are features fed into the final MLP (e.g., rolling averages)
    # target_col must already be the return on the raw scale (see
    # add_target_returns). Computing it here from a normalised frame is a bug.

    # Reconstruct continuous calendar
    min_date = df["date"].min()
    max_date = df["date"].max()
    date_grid = pl.DataFrame({"date": pl.date_range(min_date, max_date, "1d", eager=True)})

    seq_list, cross_list, y_list, cls_list, meta_list = [], [], [], [], []

    for cls_name, grp in df.group_by("target_class"):
        class_name = cls_name[0] if isinstance(cls_name, tuple) else cls_name
        grp = date_grid.join(grp, on="date", how="left").sort("date")
        # LOCF only the model inputs. Targets must stay null on non-trading /
        # non-published dates so that ``valid_mask`` can drop them; forward-
        # filling targets silently replaces them with stale returns.
        fill_cols = [c for c in (seq_cols + cross_cols) if c in grp.columns]
        if fill_cols:
            grp = grp.with_columns([pl.col(c).forward_fill() for c in fill_cols])

        valid_dates = set(df.filter(pl.col("target_class") == class_name)["date"].to_list())

        seq_arr = grp.select(seq_cols).to_torch()

        if cross_cols:
            cross_arr = grp.select(cross_cols).to_torch()
        else:
            cross_arr = torch.empty((grp.height, 0))

        tgt_name = target_col or f"target_ret_h{h}"
        if tgt_name in grp.columns:
            y_arr = grp[tgt_name].to_torch()
        elif f"y_step_h{h}" in grp.columns:
            # Fallback for callers that did not pre-compute returns. Only
            # correct when ``log_value`` has not been standardised.
            y_arr = (grp[f"y_step_h{h}"] - grp["log_value"]).to_torch()
        else:
            y_arr = torch.full((grp.height,), float('nan'))

        # Explicit NaNs so target nulls survive the .to_torch() conversion and
        # can be filtered by the training/eval valid_mask.
        if y_arr.dtype.is_floating_point:
            y_arr = y_arr.to(torch.float32)

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


class _StallWatchdog:
    """Abort the process if no heartbeat arrives within ``timeout`` seconds.

    A hard CUDA stall blocks the training thread inside a kernel and cannot be
    interrupted from Python, so the only reliable escape is process exit. Every
    epoch is checkpointed, so the caller can resume from the latest checkpoint.
    """

    def __init__(self, timeout: float, label: str) -> None:
        self.timeout = timeout
        self.label = label
        self._last = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def heartbeat(self) -> None:
        self._last = time.monotonic()

    def _run(self) -> None:
        while not self._stop.wait(5.0):
            if time.monotonic() - self._last > self.timeout:
                LOGGER.critical(
                    "stall timeout (%.0fs) exceeded for %s; aborting process so "
                    "the last checkpoint can be resumed",
                    self.timeout, self.label,
                )
                os._exit(3)

    def __enter__(self) -> Self:
        if self.timeout > 0:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _checkpoint_path(h: int, seed: int) -> Path:
    return CHECKPOINT_DIR / f"lstm_h{h}_seed{seed}.pt"


def _save_checkpoint(
    h: int,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    best_state: dict[str, torch.Tensor] | None,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "best_state": best_state,
        },
        _checkpoint_path(h, seed),
    )


class LSTMModel:
    def __init__(
        self,
        max_epochs: int = 100,
        patience: int = 8,
        hidden: int = 32,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        resume: bool = False,
        checkpoint: bool = True,
    ) -> None:
        self.max_epochs = max_epochs
        self.patience = patience
        self.hidden = hidden
        self.seeds = seeds
        self.resume = resume
        self.checkpoint = checkpoint
        self.models: dict[int, list[tuple[int, QuantileLSTM]]] = {}
        self.target_stats: dict[int, dict[str, tuple[float, float]]] = {}

    def fit(self, train: pl.DataFrame, horizons: list[int]) -> None:
        self.horizons = horizons
        self.models = {}

        self._fit_scalers(train)

        train_aug = add_target_returns(train, horizons)
        train_norm = self._preprocess(train_aug)

        for h in horizons:
            self.target_stats[h] = target_stats(train, h)
            low_w, high_w = standardized_halfwidths(train, h)

            seq_all, cross_all, c_all, y_all, meta_all = create_windows(
                train_norm, h, self.seq_cols, self.cross_cols, target_col=f"target_ret_h{h}"
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

            # Standardise the raw return per class using train statistics only.
            mu = torch.tensor([self.target_stats[h][m[1]][0] for m in meta_all])
            sd = torch.tensor([self.target_stats[h][m[1]][1] for m in meta_all])
            y_all = (y_all - mu) / sd

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
            LOGGER.info("Training on device: %s for h=%d (%d seeds)", device, h, len(self.seeds))

            seed_models: list[tuple[int, QuantileLSTM]] = []
            for seed in self.seeds:
                set_seed(seed)
                model = self._train_one(
                    seed, h, device, seq_tr, cross_tr, c_tr, y_tr,
                    seq_es, cross_es, c_es, y_es, low_w, high_w,
                )
                seed_models.append((seed, model))
            self.models[h] = seed_models

    def _train_one(
        self,
        seed: int,
        h: int,
        device: torch.device,
        seq_tr: torch.Tensor,
        cross_tr: torch.Tensor,
        c_tr: torch.Tensor,
        y_tr: torch.Tensor,
        seq_es: torch.Tensor,
        cross_es: torch.Tensor,
        c_es: torch.Tensor,
        y_es: torch.Tensor,
        low_w: float,
        high_w: float,
    ) -> QuantileLSTM:
        model = QuantileLSTM(
            n_seq_features=len(self.seq_cols),
            n_cross_features=len(self.cross_cols),
            hidden=self.hidden,
        )
        model.init_quantile_biases(low_w, high_w)
        model = model.to(device)

        lr = 1e-4 if h >= 30 else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        tr_ds = TensorDataset(seq_tr, cross_tr, c_tr, y_tr)
        tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)

        if len(seq_es) > 0:
            seq_es = seq_es.to(device)
            cross_es = cross_es.to(device)
            c_es = c_es.to(device)
            y_es = y_es.to(device)

        start_epoch = 0
        best_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None

        ckpt = _checkpoint_path(h, seed)
        if self.resume and ckpt.exists():
            try:
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                start_epoch = int(state["epoch"]) + 1
                best_loss = float(state["best_loss"])
                best_state = state.get("best_state")
                LOGGER.info("resumed h=%d seed=%d from epoch %d", h, seed, start_epoch)
            except Exception as exc:  # noqa: BLE001 - a corrupt checkpoint must not kill the run
                LOGGER.warning("could not resume h=%d seed=%d: %s", h, seed, exc)

        label = f"lstm h={h} seed={seed}"
        watchdog = _StallWatchdog(STALL_TIMEOUT_S, label)
        with watchdog:
            for epoch in range(start_epoch, self.max_epochs):
                model.train()
                for b_seq, b_cross, bc, by in tr_dl:
                    b_seq = b_seq.to(device)
                    b_cross = b_cross.to(device)
                    bc = bc.to(device)
                    by = by.to(device)
                    optimizer.zero_grad()
                    pred = model(b_seq, b_cross, bc)
                    loss = huber_pinball_loss(pred, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    watchdog.heartbeat()

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

                if self.checkpoint:
                    _save_checkpoint(h, seed, model, optimizer, epoch, best_loss, best_state)

                if patience_counter >= self.patience:
                    LOGGER.info("h=%d seed=%d early stop at epoch %d", h, seed, epoch)
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def _fit_scalers(self, train: pl.DataFrame) -> None:
        """Select input columns and compute their standardisation stats.

        Target columns are excluded from ``EXCLUDE`` so they never become model
        inputs. ``log_value`` is the only sequence feature; everything else is a
        cross-sectional feature fed to the head.
        """
        all_feature_cols = [c for c in train.columns if c not in EXCLUDE]
        self.seq_cols = ["log_value"]
        self.cross_cols = [c for c in all_feature_cols if c not in self.seq_cols]
        all_cols_to_scale = self.seq_cols + self.cross_cols

        mu_df = train.select([pl.col(c).mean().alias(c) for c in all_cols_to_scale])
        sd_df = train.select([pl.col(c).std().alias(c) for c in all_cols_to_scale])

        self.mu = {c: mu_df[c][0] for c in all_cols_to_scale}
        self.sd = {c: sd_df[c][0] for c in all_cols_to_scale}

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

    def _predict_windows(
        self, df: pl.DataFrame, h: int
    ) -> tuple[list[tuple[object, str]], torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        df_aug = add_target_returns(df, [h])
        df_norm = self._preprocess(df_aug)
        seq_eval, cross_eval, c_eval, _, meta_eval = create_windows(
            df_norm, h, self.seq_cols, self.cross_cols, target_col=f"target_ret_h{h}"
        )
        if len(seq_eval) > 0:
            seq_eval = seq_eval.to(device)
            cross_eval = cross_eval.to(device)
            c_eval = c_eval.to(device)
        return meta_eval, seq_eval, cross_eval, c_eval

    def _invert(self, preds: torch.Tensor, meta_eval: list[tuple[object, str]], h: int) -> tuple[list[float], list[float], list[float]]:
        """Undo standardisation, returning p10/p50/p90 in return space."""
        mu = np.array([self.target_stats[h][m[1]][0] for m in meta_eval], dtype=np.float64)
        sd = np.array([self.target_stats[h][m[1]][1] for m in meta_eval], dtype=np.float64)
        arr = preds.numpy().astype(np.float64)
        arr = arr * sd[:, None] + mu[:, None]
        return arr[:, 0].tolist(), arr[:, 1].tolist(), arr[:, 2].tolist()

    def predict_per_seed(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, dict[int, pl.DataFrame]]]:
        """Raw per-seed quantile frames: ``[split][h][seed]``.

        Exposed so callers can report single-seed spread and can checkpoint the
        individual seed predictions rather than only the ensemble mean.
        """
        out: dict[str, dict[int, dict[int, pl.DataFrame]]] = {
            name: {} for name in eval_frames
        }
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for h in self.horizons:
            for split_name, df in eval_frames.items():
                meta_eval, seq_eval, cross_eval, c_eval = self._predict_windows(df, h)
                out[split_name][h] = {}
                if len(seq_eval) == 0:
                    continue
                for seed, model in self.models[h]:
                    model.to(device)
                    with torch.no_grad():
                        preds = model(seq_eval, cross_eval, c_eval).cpu()
                    p10r, p50r, p90r = self._invert(preds, meta_eval, h)
                    pred_df = pl.DataFrame({
                        "date": [m[0] for m in meta_eval],
                        "target_class": [m[1] for m in meta_eval],
                        "p_0.1_ret": p10r,
                        "p_0.5_ret": p50r,
                        "p_0.9_ret": p90r,
                    })
                    res = self._to_levels(df, pred_df, h)
                    out[split_name][h][seed] = res
        return out

    def _to_levels(self, df: pl.DataFrame, pred_df: pl.DataFrame, h: int) -> pl.DataFrame:
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
        return res.drop_nulls(["p_0.5"])

    def predict(
        self, eval_frames: dict[str, pl.DataFrame]
    ) -> dict[str, dict[int, pl.DataFrame]]:
        """Ensemble-mean prediction: mean of the per-seed quantiles per row."""
        per_seed = self.predict_per_seed(eval_frames)
        out: dict[str, dict[int, pl.DataFrame]] = {
            name: {} for name in eval_frames
        }
        for split_name, h_map in per_seed.items():
            for h, seed_map in h_map.items():
                if not seed_map:
                    out[split_name][h] = pl.DataFrame(
                        schema={
                            "date": pl.Date,
                            "target_class": pl.String,
                            "h": pl.Int64,
                            "p_0.1": pl.Float64,
                            "p_0.5": pl.Float64,
                            "p_0.9": pl.Float64,
                        }
                    )
                    continue
                # Average across seeds on the row identity; all seeds share the
                # same meta/window order, so a horizontal mean is exact.
                frames = list(seed_map.values())
                base = frames[0].select("date", "target_class", "h")
                for q in ("p_0.1", "p_0.5", "p_0.9"):
                    vals = np.mean(
                        np.stack([f[q].to_numpy() for f in frames], axis=0), axis=0
                    )
                    base = base.with_columns(pl.Series(q, vals))
                out[split_name][h] = base
        return out
