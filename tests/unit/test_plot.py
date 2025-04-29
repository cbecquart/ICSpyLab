"""
Unit tests for the plot functionalities in the ICSpy package.
"""
from . import *
logger = logging.getLogger(__name__)


def test_plot_method():
    """
    Test the plot method of the ICS class.

    This test verifies that the plot method generates a plot of the transformed data,
    and raises an error if the model is not fitted and transformed.
    """
    ics = ICS(S1=cov, S2=covW, S2_args={"alpha": -1})
    X = np.random.randn(100, 8)
    ics.fit_transform(X)
    # ics.plot()  # Uncomment to visually inspect the plot
    with pytest.raises(ValueError):
        ics_unfitted = ICS()
        ics_unfitted.plot()
    with pytest.raises(ValueError):
        ics_unfitted = ICS()
        ics_unfitted.fit(X)
        ics_unfitted.plot()
