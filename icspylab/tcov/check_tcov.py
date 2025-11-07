import numpy as np
import warnings
try:
    import tcov_module
except ImportError:
    tcov_module = None


if __name__ == "__main__":
    X = np.random.randn(100, 5)
    if tcov_module is None:
        raise ImportError('Requires tcov_module which is not available. For help building the module, see '
                          'icspylab/tcov/README.md. Proceeding with use_cpp=False.')
    beta = float(2)
    tcov_X = tcov_module.tcov_cpp(X, beta)
    print('tcov_X =', tcov_X)
    print("tcov_cpp successfully implemented.")
