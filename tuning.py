import yfinance as yf

from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
from keras.api.optimizers import Adam

import keras_tuner as kt
import tensorflow as tf

from utility import min_max_scale, inverse_scale, create_dataset, split_train_test

TEST_SIZE = 0.05


def build_lstm_model(hp, input_shape):
    """
    Build an LSTM model with hyperparameters tuned by Keras Tuner.

    Args:
        hp (HyperParameters): Keras Tuner hyperparameter object.
        input_shape (tuple): Shape of the input data.

    Returns:
        Sequential: Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))

    model.add(LSTM(units=hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True))
    model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(units=hp.Int('dense_units', min_value=20, max_value=80, step=20), activation='relu'))
    model.add(Dense(1))

    learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-4])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

    return model


def tune_hyperparameters(ticker, look_back, max_trials, executions_per_trial, epochs, verbose=True):
    """
    Tune hyperparameters for the LSTM model but do not train the final model.

    Args:
        ticker (str): Stock ticker symbol.
        look_back (int): Number of previous time steps to use as input features.
        max_trials (int): Maximum number of hyperparameter combinations to try.
        executions_per_trial (int): Number of models to build for each trial.
        epochs (int): Number of epochs to train during hyperparameter search.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: (tuner, (data_splits, scaling_info)) where:
            tuner: The tuner object with best hyperparameters found
            data_splits: Tuple of (X_train, y_train, X_test, y_test)
            scaling_info: Tuple of (min_val, max_val)
    """
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01", progress=verbose)
    data = data[['Close']].dropna()

    normalized_data, min_val, max_val = min_max_scale(data.values)

    X, y = create_dataset(normalized_data, look_back)

    X_train, y_train, X_test, y_test = split_train_test(X, y, TEST_SIZE)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, input_shape=(look_back, 1)),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='hyper_tuning',
        project_name=f'{ticker}_lstm_tuning'
    )

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        verbose=verbose
    )

    data_info = ((X_train, y_train, X_test, y_test), (min_val, max_val))
    return tuner, data_info


def train_best_model(tuner, data_splits, scaling_info, look_back, epochs, verbose=True):
    """
    Train the best model found by the tuner with the specified look_back and epochs.

    Args:
        tuner (RandomSearch): Keras Tuner object with best hyperparameters.
        data_splits (tuple): Tuple of (X_train, y_train, X_test, y_test).
        scaling_info (tuple): Tuple of (min_val, max_val) for inverse scaling.
        look_back (int): Number of previous time steps to use as input features.
        epochs (int): Number of training epochs.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: (normalized_rmse, original_rmse, model, predictions) where:
            normalized_rmse: RMSE calculated using normalized values
            original_rmse: RMSE calculated using original scale values
            model: Trained Keras model
            predictions: Tuple of (train_preds_original, test_preds_original, y_train_original, y_test_original)
    """
    X_train, y_train, X_test, y_test = data_splits
    min_val, max_val = scaling_info

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = build_lstm_model(best_hp, input_shape=(look_back, 1))

    batch_size = int(len(X_train) * 0.01)  # Using a fixed batch fraction of 0.01
    batch_size = max(1, batch_size)

    if verbose:
        print("Best Hyperparameters:", best_hp.values)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    train_preds = model.predict(X_train, verbose=verbose)
    test_preds = model.predict(X_test, verbose=verbose)

    rmse_normalized = tf.sqrt(tf.reduce_mean(tf.square(test_preds - y_test))).numpy()

    train_preds_original = inverse_scale(train_preds, min_val, max_val)
    test_preds_original = inverse_scale(test_preds, min_val, max_val)
    y_train_original = inverse_scale(y_train, min_val, max_val)
    y_test_original = inverse_scale(y_test, min_val, max_val)

    rmse_original = tf.sqrt(tf.reduce_mean(tf.square(test_preds_original - y_test_original))).numpy()

    if verbose:
        print(f"Normalized RMSE: {rmse_normalized}")
        print(f"Original RMSE: {rmse_original}")

    return rmse_normalized, rmse_original, model, (train_preds_original, test_preds_original, y_train_original, y_test_original)