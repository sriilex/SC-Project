"""Data-domain and test-isolation regression for the compact experiment loader."""

from datetime import date

import numpy as np
import polars as pl

from ml import compact_sugeno_experiment as experiment


def test_validation_loader_excludes_nonpositive_history_and_future(
    tmp_path, monkeypatch
):
    data = tmp_path / "src/data"
    (data / "preds").mkdir(parents=True)
    dates = [date(2020, 1, 1), date(2020, 1, 3)]
    source = pl.DataFrame(
        {
            "date": dates,
            "target_class": ["Capesize"] * 2,
            "log_value": np.log([100.0, 101.0]),
            "y_step_h1": [4.6, 4.7],
        }
    )
    source.write_parquet(data / "samples_valid.parquet")
    pred = source.select("date", "target_class").with_columns(
        pl.lit(1).alias("h"),
        pl.lit(4.0).alias("p_0.1"),
        pl.lit(4.5).alias("p_0.5"),
        pl.lit(5.0).alias("p_0.9"),
    )
    pred.write_parquet(data / "preds/lstm_fixture_valid_h1.parquet")
    pl.DataFrame(
        {
            "date": [date(2020, 1, i) for i in (1, 2, 3, 4)],
            "series_id": ["BC_INDEX"] * 4,
            "value": [100.0, -100.0, 101.0, 99999.0],
        }
    ).write_parquet(data / "master_long.parquet")
    # A test read would fail immediately: this is intentionally not a parquet.
    (data / "samples_test.parquet").write_text("DO NOT READ", encoding="utf-8")
    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(experiment, "HORIZONS", (1,))
    contexts, daily, hashes = experiment.load_validation("fixture")
    assert contexts[1].height == 2
    assert daily["date"].to_list() == dates
    assert np.isfinite(daily["log_value"].to_numpy()).all()
    assert all(len(h) == 64 for h in hashes.values())
