import numpy as np
import tensorflow as tf

def min_max_scale(data):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)

    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data, min_val, max_val

def inverse_scale(data, min_val, max_val):
    return data * (max_val - min_val) + min_val

def create_dataset(dataset, look_back):
    X, Y = [], []

    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])

    return np.array(X), np.array(Y)

def split_train_test(X, y, test_ratio):
    test_size = int(len(X) * test_ratio)
    train_size = len(X) - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test
