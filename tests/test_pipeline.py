import json
from pathlib import Path

import pytest

from sleeprisk.cli import main
from sleeprisk.config import apply_overrides, load_config
from sleeprisk.pipeline import run


def test_config_coerces_yaml_scientific_notation():
    cfg = apply_overrides({"a": {"lr": 1}}, ["a.lr=1e-5", "a.flag=true", "a.list=[x,y]"])
    assert cfg["a"]["lr"] == pytest.approx(1e-5) and isinstance(cfg["a"]["lr"], float)
    assert cfg["a"]["flag"] is True and cfg["a"]["list"] == ["x", "y"]
    assert isinstance(load_config()["models"]["lstm"]["weight_decay"], float)


def test_end_to_end_writes_all_artifacts(cfg):
    metrics = run(cfg)
    out = Path(cfg["data"]["output_dir"])
    for f in ["metrics.json", "report.md", "user_predictions.csv", "config_used.yaml", "logistic_coefficients.csv"]:
        assert (out / f).exists(), f
    assert set(metrics["subject_cv"]) == {"majority", "logistic_user", "random_forest", "lstm"}
    assert all(r["test_windows_from_seen_users_pct"] > 90 for r in metrics["leaky_replica"])
    json.dumps(metrics, default=str)


def test_cli_runs(cfg, synth_dir, tmp_path):
    rc = main(
        [
            "run",
            "--data-dir",
            str(synth_dir),
            "--out",
            str(tmp_path / "cli"),
            "--set",
            "labels.threshold=8",
            "--set",
            "cv.n_splits=3",
            "--set",
            "cv.n_repeats=1",
            "--set",
            "cv.n_permutations=0",
            "--set",
            "models.run=[majority,logistic_user]",
        ]
    )
    assert rc == 0 and (tmp_path / "cli" / "report.md").exists()
