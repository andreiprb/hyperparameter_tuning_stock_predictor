from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
from keras.api.optimizers import Adam

import keras_tuner as kt
import tensorflow as tf

from utility import inverse_scale, prepare_stock_data
from constants import TEST_SIZE, BATCH_FRACTION


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


def tune_hyperparameters(ticker=None, look_back=None, max_trials=None, executions_per_trial=None,
                         epochs=None, verbose=True, prepared_data=None):
    """
    Tune hyperparameters for the LSTM model but do not train the final model.

    Args:
        ticker (str, optional): Stock ticker symbol. Not used if prepared_data is provided.
        look_back (int, optional): Number of previous time steps. Required if ticker is provided.
        max_trials (int): Maximum number of hyperparameter combinations to try.
        executions_per_trial (int): Number of models to build for each trial.
        epochs (int): Number of epochs to train during hyperparameter search.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
        prepared_data (dict, optional): Pre-prepared data to use instead of downloading. If provided,
                                     ticker and look_back are ignored.

    Returns:
        tuple: (tuner, (data_splits, scaling_info))
    """
    if prepared_data is None:
        prepared_data = prepare_stock_data(
            ticker=ticker,
            start_date="2020-01-01",
            end_date="2023-01-01",
            look_back=look_back,
            test_ratio=TEST_SIZE,
            verbose=verbose
        )

        if prepared_data is None:
            raise ValueError(f"Insufficient data for ticker {ticker}")

    X_train, y_train, X_test, y_test = prepared_data['data_splits']
    scaling_info = prepared_data['scaling_info']

    input_shape = (X_train.shape[1], X_train.shape[2])

    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, input_shape),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='hyper_tuning',
        project_name=f'{ticker or "_combined"}_lstm_tuning',
    )

    tuner.search(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        verbose=verbose
    )

    data_info = ((X_train, y_train, X_test, y_test), scaling_info)
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

    batch_size = int(len(X_train) * BATCH_FRACTION)
    batch_size = max(1, batch_size)

    if verbose:
        print("Best Hyperparameters:", best_hp.values)
        print(f"Using batch size of {batch_size} (fraction: {BATCH_FRACTION})")

    model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=TEST_SIZE,
        verbose=verbose
    )

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

    return rmse_normalized, rmse_original, model, (
    train_preds_original, test_preds_original, y_train_original, y_test_original)


def evaluate_model(model, data_splits, scaling_info, verbose=False):
    """
    Evaluate a model on given data splits.

    Args:
        model: Trained Keras model
        data_splits: Tuple of (X_train, y_train, X_test, y_test)
        scaling_info: Tuple of (min_val, max_val) for inverse scaling
        verbose: Whether to print progress

    Returns:
        tuple: (normalized_rmse, original_rmse, predictions)
    """
    X_train, y_train, X_test, y_test = data_splits
    min_val, max_val = scaling_info

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

    predictions = (train_preds_original, test_preds_original, y_train_original, y_test_original)

    return rmse_normalized, rmse_original, predictions