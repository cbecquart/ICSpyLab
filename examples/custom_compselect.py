import pandas as pd
from icspylab import ICS, ComponentSelect, plot_ics
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Instantiate ICS object
ics = ICS(S1="cov", S2="cov4", algorithm="standard")

# Fit and transform the ICS model
X_ics = ics.fit_transform(X)
plot_ics(X_ics)

# Keep the last component only
X_ics_reduced = X_ics[:, -1]
print("Shape after ICS and manual component selection:", X_ics_reduced.shape)


## Create custom component selection method

def select_last_comp(W, **kwargs):
    all_comp_names = [f"IC_{i + 1}" for i in range(W.shape[1])]
    p = W.shape[1]
    selected_component_names = all_comp_names[-1:]

    # Keep only the selected components
    name_to_idx = {name: i for i, name in enumerate(all_comp_names)}
    idx = [name_to_idx[name] for name in selected_component_names]
    components = W[idx, :]
    n_components = len(selected_component_names)

    return ComponentSelect(label="custom", components=components, n_components=n_components,
                           component_names=selected_component_names, info=None)

# Instantiate ICS object
ics_custom = ICS(S1="cov", S2="cov4", algorithm="standard", method_select=select_last_comp)

# Fit and transform the ICS model
X_ics_custom = ics_custom.fit_transform(X)

print(f"Shape after ICS with select_last_comp: {X_ics_custom.shape}"
      f" with component names: {ics_custom.component_names_}")


## Update function

def select_last_q_comp(W, q=1, **kwargs):
    all_comp_names = [f"IC_{i + 1}" for i in range(W.shape[1])]
    p = W.shape[1]
    selected_component_names = all_comp_names[-q:]

    # Keep only the selected components
    name_to_idx = {name: i for i, name in enumerate(all_comp_names)}
    idx = [name_to_idx[name] for name in selected_component_names]
    components = W[idx, :]
    n_components = len(selected_component_names)

    return ComponentSelect(label="custom", components=components, n_components=n_components,
                           component_names=selected_component_names, info=None)

# Instantiate ICS object with select_last_q_comp and default parameters
ics_custom = ICS(S1="cov", S2="cov4", algorithm="standard", method_select=select_last_q_comp)

# Fit and transform the ICS model
X_ics_custom = ics_custom.fit_transform(X)

print(f"Shape after ICS with select_last_q_comp (default q): {X_ics_custom.shape}"
      f" with component names: {ics_custom.component_names_}")

# Instantiate ICS object with select_last_q_comp and q=2
ics_custom = ICS(S1="cov", S2="cov4", algorithm="standard", method_select=select_last_q_comp, select_args={"q": 2})

# Fit and transform the ICS model
X_ics_custom = ics_custom.fit_transform(X)

print(f"Shape after ICS with select_last_q_comp (q=2): {X_ics_custom.shape}"
      f" with component names: {ics_custom.component_names_}")
