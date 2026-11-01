import pytest

from sleeprisk.config import load_config
from sleeprisk.synthetic import write_synthetic_studentlife


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    return write_synthetic_studentlife(tmp_path_factory.mktemp("sl") / "dataset", n_users=18, days=30, seed=3)


@pytest.fixture()
def cfg(synth_dir, tmp_path):
    return load_config(
        overrides=[
            f"data.base_dir={synth_dir}",
            f"data.output_dir={tmp_path / 'results'}",
            "labels.threshold=8",
            "cv.n_splits=3",
            "cv.n_repeats=1",
            "cv.n_permutations=3",
            "cv.bootstrap=50",
            "models.lstm.max_epochs=3",
        ]
    )
