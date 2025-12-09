import numpy as np
from scipy.stats import shapiro, normaltest, jarque_bera

def _agostino_test(x):
    """D’Agostino K² test."""
    stat, p = normaltest(x)
    return {"statistic": stat, "p_value": p}

def _jarque_test(x):
    """Jarque–Bera test."""
    stat, p = jarque_bera(x)
    return {"statistic": stat, "p_value": p}

def _shapiro_test(x):
    """Shapiro test."""
    stat, p = shapiro(x)
    return {"statistic": stat, "p_value": p}

def normal_crit(X, level=0.05,
                test="agostino_test",
                max_select=None):

    # Choix du test
    tests = {
        "agostino_test": _agostino_test,
        "jarque_test": _jarque_test,
        "shapiro_test": _shapiro_test
    }

    if test not in tests:
        valid = ", ".join(tests.keys())
        raise ValueError(f"test must be one of: {valid}")

    test_fun = tests[test]

    # Initialization
    comp_select = [f"IC_{i+1}" for i in range(X.shape[1])]
    if max_select is None:
        max_select = X.shape[1] - 1

    # Apply marginal normality tests to all components and select on p-values
    test_pvals = np.array([test_fun(X[:, j])["p_value"] for j in range(X.shape[1])])

    out = {
        "crit": "normal",
        "level": level,
        "max_select": max_select,
        "test": test,
        "pvalues": test_pvals.copy()
    }

    comp_signif = test_pvals <= level

    # If none of them are significative
    if comp_signif.sum() == 0:
        select = []
        adjusted_levels = [level]

    # Else: we select the components on the extreme left and right while they are not gaussian
    else:
        temp = 1
        select = []
        adjusted_levels = [level]

        pvals = test_pvals.copy()
        comps = comp_select.copy()

        while temp <= max_select:

            left_pval = pvals[0]
            right_pval = pvals[-1]

            if (left_pval < level) and (left_pval < right_pval):
                select.append(comps[0])
                comps = comps[1:]
                pvals = pvals[1:]

            elif right_pval < level:
                select.append(comps[-1])
                comps = comps[:-1]
                pvals = pvals[:-1]

            else:
                break

            temp += 1

            # Adjust the alpha level by dividing by the old weight and multiplying by the new one.
            pvals = pvals * temp / (temp - 1)
            adjusted_levels.append(level / temp)

    out.update({
        "adjusted_levels": adjusted_levels,
        "select": select
    })

    return out
