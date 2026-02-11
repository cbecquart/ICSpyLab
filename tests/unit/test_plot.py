"""
Unit tests for the plot functionalities in the ICSpyLab package.
"""

import logging
import pytest
import numpy as np
from icspylab import plot_ics
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def test_plot_ics_invalid_input_raises():
    with pytest.raises(ValueError):
        plot_ics([1, 2, 3])


def test_plot_ics_uses_all_components(monkeypatch):
    X = np.random.randn(10, 4)

    called = {}

    def fake_pairplot(df, **kwargs):
        called["n_cols"] = df.shape[1]

    monkeypatch.setattr(sns, "pairplot", fake_pairplot)
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_ics(X)

    assert called["n_cols"] == 4


def test_plot_ics_uses_first_and_last_components(monkeypatch):
    X = np.random.randn(10, 10)

    called = {}

    def fake_pairplot(df, **kwargs):
        called["columns"] = list(df.columns)

    monkeypatch.setattr(sns, "pairplot", fake_pairplot)
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_ics(X)

    assert called["columns"] == [
        "IC_1", "IC_2", "IC_3",
        "IC_8", "IC_9", "IC_10"
    ]


def test_plot_ics_forwards_kwargs(monkeypatch):
    X = np.random.randn(10, 3)

    def fake_pairplot(df, **kwargs):
        assert kwargs["corner"] is True

    monkeypatch.setattr(sns, "pairplot", fake_pairplot)
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_ics(X, corner=True)
