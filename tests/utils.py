import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes

def calculate_similarity_percentage(r_matrix, py_matrix):
    """
    Calculate the percentage similarity between two matrices.

    Parameters:
        r_matrix (numpy.ndarray): The first matrix.
        py_matrix (numpy.ndarray): The second matrix.

    Returns:
        result (numpy.ndarray): The percentage difference matrix.
    """
    matrix1 = np.array(r_matrix)
    matrix2 = np.array(py_matrix)

    # Ensure matrices have the same shape
    if r_matrix.shape != py_matrix.shape:
        raise ValueError("Matrices must have the same shape.")

    # Calculate the element-wise relative difference
    relative_difference = np.abs(r_matrix - py_matrix) / np.abs(r_matrix)

    # Convert the relative difference to a similarity percentage
    similarity_percentage = (1 - relative_difference) * 100

    return similarity_percentage



def load_dataset(dataset_name):
    """
    Load the dataset by name.

    This function loads the specified dataset using scikit-learn's dataset loaders.

    Parameters:
        dataset_name (str): The name of the dataset to load. Supported values are
                            'iris', 'wine', 'breast_cancer', 'diabetes'.

    Returns:
        tuple: A tuple containing the data matrix and the target vector.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    return data.data, data.target