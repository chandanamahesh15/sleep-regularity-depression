"""Classical models. Every preprocessing step lives *inside* the sklearn Pipeline,
so imputation/scaling statistics are always fit on the training fold only."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_majority(seed: int = 0) -> DummyClassifier:
    """Predicts the training-fold prevalence. Any real model must beat this."""
    return DummyClassifier(strategy="prior", random_state=seed)


def make_logistic(C: float = 0.5, seed: int = 0) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)),
        ]
    )


def make_random_forest(
    n_estimators: int = 300, max_depth: int = 6, min_samples_leaf: int = 5, seed: int = 0
) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
