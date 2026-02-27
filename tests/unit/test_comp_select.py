import pytest
import numpy as np
from icspylab.comp_select import ComponentSelect, normal_crit, med_crit, _validate_nb_select

def test_component_select_init():
    """Test that ComponentSelect initializes correctly."""
    comps = np.eye(2)
    cs = ComponentSelect(label="test", components=comps, n_components=2,
                         component_names=["IC_1", "IC_2"], info=None)
    assert cs.label == "test"
    np.testing.assert_array_equal(cs.components, comps)
    assert cs.n_components == 2
    assert cs.component_names == ["IC_1", "IC_2"]
    assert cs.info is None

def test_normal_crit_basic():
    """Test that normal_crit returns a ComponentSelect with correct info."""
    X = np.random.randn(10, 3)
    W = np.eye(3)
    res = normal_crit(X, W, test="agostino", level=0.05)
    assert isinstance(res, ComponentSelect)
    assert hasattr(res, "components")
    assert hasattr(res, "info")
    assert "pvalues" in res.info
    assert len(res.info["pvalues"]) == X.shape[1]

def test_med_crit_basic():
    """Test that med_crit returns a ComponentSelect with correct selection."""
    kurtosis = np.array([1.0, 2.0, 3.0])
    W = np.eye(3)
    res = med_crit(kurtosis, W, nb_select=2)
    assert isinstance(res, ComponentSelect)
    assert res.n_components == 2
    assert len(res.component_names) == 2
    # Check that selected components are valid names
    assert all(name.startswith("IC_") for name in res.component_names)

@pytest.mark.parametrize("nb_select, p, expected", [
    (None, 5, 4),
    (2, 5, 2)
])
def test_validate_nb_select_valid(nb_select, p, expected):
    """Test _validate_nb_select with valid input."""
    assert _validate_nb_select(nb_select, p) == expected

@pytest.mark.parametrize("nb_select", ["2", 0, -1, 5])
def test_validate_nb_select_invalid(nb_select):
    """Test _validate_nb_select raises errors with invalid input."""
    with pytest.raises((TypeError, ValueError)):
        _validate_nb_select(nb_select, 5)
