import numpy as np

def one_vs_all(Y, target_digit):
    """
    Converts multi-class labels into binary labels for One-vs-All classification.
    
    Args:
        Y (numpy.ndarray): Original labels.
        target_digit (int): The class to be assigned +1 (all others get -1).
    
    Returns:
        numpy.ndarray: Transformed binary labels.
    """
    return np.where(Y == target_digit, 1, -1)

def one_vs_one(X, Y, class_1, class_2):
    """
    Filters dataset for two specific classes and converts labels for One-vs-One classification.
    
    Args:
        X (numpy.ndarray): Feature matrix.
        Y (numpy.ndarray): Original labels.
        class_1 (int): First class (assigned +1).
        class_2 (int): Second class (assigned -1).
    
    Returns:
        tuple: (Filtered X, Transformed Y, Indices in original dataset)
    """
    mask = (Y == class_1) | (Y == class_2)
    indices = np.where(mask)[0]  # indices in the original dataset
    X_filtered = X[mask]
    Y_filtered = np.where(Y[mask] == class_1, 1, -1)
    
    return X_filtered, Y_filtered, indices