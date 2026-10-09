"""Testnivå 3: modell. Kontrakt och reproducerbarhet."""

import json
from pathlib import Path

import numpy as np
import pytest

from churn.data import FEATURES, dela_upp, las_data, skapa_features
from churn.model import trana_och_utvardera, SEED

from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def resultat():
    return trana_och_utvardera()


def test_modellkontrakt(resultat):
    modell, _ = resultat
    X, _ = dela_upp(skapa_features(las_data().head(5)))
    assert list(X.columns) == FEATURES
    sannolikhet = modell.predict_proba(X)
    assert sannolikhet.shape == (5, 2)
    assert np.all((sannolikhet >= 0) & (sannolikhet <= 1))


def test_reproducerbar(resultat):
    _, matvarden = resultat
    _, igen = trana_och_utvardera()
    assert igen == matvarden


def test_modellens_roc_auc(resultat):
    _, matvarden = resultat
    assert matvarden["roc_auc"] >= 0.70


def test_modellen_slar_dummy(resultat):
    _, matvarden = resultat

    X, y = dela_upp(skapa_features(las_data()))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)

    sannolikhet = dummy.predict_proba(X_test)[:, 1]
    dummy_auc = roc_auc_score(y_test, sannolikhet)

    assert matvarden["roc_auc"] > dummy_auc
