"""Isolated sample building must not overwrite canonical/test artifacts."""

from datetime import date

import polars as pl

from ml.datasets import build_samples


def test_custom_sample_config_can_skip_test(tmp_path, monkeypatch):
    dates = pl.date_range(date(2018, 1, 1), date(2020, 1, 1), "1d", eager=True)
    frames = []
    for i, series in enumerate(("BC_INDEX", "BPI_INDEX", "BSI_INDEX", "BD_INDEX")):
        frames.append(
            pl.DataFrame(
                {
                    "date": dates,
                    "series_id": [series] * len(dates),
                    "value": [100.0 + i] * len(dates),
                }
            ).filter(pl.col("date").dt.weekday() <= 5)
        )
    pl.concat(frames).write_parquet(tmp_path / "master.parquet")
    config = {
        "paths": {
            "master_parquet": "master.parquet",
            "samples_train": "isolated/train.parquet",
            "samples_valid": "isolated/valid.parquet",
            "samples_test": "DO_NOT_CREATE.parquet",
            "samples_manifest": "isolated/manifest.csv",
        },
        "targets": {
            "classes": {
                "BC_INDEX": "Capesize",
                "BPI_INDEX": "Panamax",
                "BSI_INDEX": "Supramax",
            },
            "horizons": {"days": [1, 7]},
        },
        "features": {
            "lags": [1, 7],
            "windows": [7],
            "returns": [1, 7],
            "bdi_col": "BD_INDEX",
            "ports": {"origin": [], "dest": []},
        },
        "calendar": {"cny_dates": []},
        "splits": {
            "valid_start": "2019-01-01",
            "test_start": "2020-01-01",
            "embargo_days": 160,
        },
    }
    monkeypatch.setattr(build_samples, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(build_samples, "load_config", lambda path: config)
    build_samples.main(tmp_path / "config.toml", include_test=False)
    assert not (tmp_path / "DO_NOT_CREATE.parquet").exists()
    assert not (tmp_path / "src/data/samples_train.parquet").exists()
    manifest = pl.read_csv(tmp_path / "isolated/manifest.csv")
    assert manifest["split"].to_list() == ["train", "valid"]
    valid = pl.read_parquet(tmp_path / "isolated/valid.parquet")
    assert valid["date"].min() >= date(2019, 1, 1)
    assert valid.filter(pl.col("date").dt.weekday() > 5).is_empty()
    # Friday's h=1 target is still null: shifting precedes non-trading masking.
    assert (
        valid.filter(pl.col("date").dt.weekday() == 5)["y_step_h1"].null_count()
        == valid.filter(pl.col("date").dt.weekday() == 5).height
    )
