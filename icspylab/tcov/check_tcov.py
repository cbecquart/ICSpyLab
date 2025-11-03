import numpy as np
import warnings
try:
    import tcov_module
except ImportError:
    warnings.warn('tcov_module not available. For help building the module, see tcov/README.md.')
    tcov_module = None


if __name__ == "__main__":
    X = np.random.randn(100, 5)
    assert tcov_module is not None, 'tcov_module not available. For help building the module, see tcov/README.md.'
    beta = float(2)
    tcov_X = tcov_module.tcov_cpp(X, beta)
    print("tcov_cpp successfully implemented.")
