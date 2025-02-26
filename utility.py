import numpy as np
import tensorflow as tf


def min_max_scale(data):
    """
    Perform min-max scaling on the input data.

    Args:
        data (array): Input data to be normalized.

    Returns:
        tuple: (normalized_data, min_val, max_val) where:
            normalized_data: Data scaled to range [0,1]
            min_val: Minimum value from original data
            max_val: Maximum value from original data
    """
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data, min_val, max_val


def inverse_scale(data, min_val, max_val):
    """
    Inverse transform min-max scaled data back to original scale.

    Args:
        data (array): Normalized data to be transformed back.
        min_val (float): Minimum value from original data.
        max_val (float): Maximum value from original data.

    Returns:
        array: Data transformed back to original scale.
    """
    return data * (max_val - min_val) + min_val


def create_dataset(dataset, look_back):
    """
    Create a time series dataset with sliding window approach.

    Args:
        dataset (array): Input time series data.
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple: (X, Y) where:
            X: Input sequences of shape (samples, look_back)
            Y: Target values corresponding to the next value after each sequence
    """
    X, Y = [], []

    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])

    return np.array(X), np.array(Y)


def split_train_test(X, y, test_ratio):
    """
    Split data into training and test sets.

    Args:
        X (array): Input features.
        y (array): Target values.
        test_ratio (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) split data arrays.
    """
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test