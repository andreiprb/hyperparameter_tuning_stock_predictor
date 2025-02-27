import yfinance as yf

from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
from keras.api.optimizers import Adam

import keras_tuner as kt

from utility import *

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
        tuple: (tuner, data_info) where:
            tuner: The tuner object with best hyperparameters found
            data_info: Dictionary containing data-related information needed for training
    """
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01", progress=verbose)
    data = data[['Close']].dropna()

    normalized_data, min_val, max_val = min_max_scale(data.values)

    X, y = create_dataset(normalized_data, look_back)

    X_train, y_train, X_test, y_test = split_train_test(X, y, TEST_SIZE)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    class LSTMHyperModel(kt.HyperModel):
        """
        Hypermodel class for LSTM model, enabling batch size tuning.
        """

        def __init__(self, input_shape):
            """
            Initialize the hypermodel with a specific input shape.

            Args:
                input_shape (tuple): Shape of the input data.
            """
            self.input_shape = input_shape

        def build(self, hp):
            """
            Build the model with the given hyperparameters.

            Args:
                hp (HyperParameters): Keras Tuner hyperparameter object.

            Returns:
                Sequential: Compiled Keras model.
            """
            return build_lstm_model(hp, self.input_shape)

        def fit(self, hp, model, x, y, validation_data=None, **kwargs):
            """
            Custom fit method that includes batch size as a hyperparameter.

            Args:
                hp (HyperParameters): Keras Tuner hyperparameter object.
                model (Model): Keras model to train.
                x (array): Input training data.
                y (array): Target training data.
                validation_data (tuple, optional): Validation data tuple. Defaults to None.

            Returns:
                History: Training history.
            """
            batch_size = hp.Choice('batch_size', values=[16, 32, 64, 128, 256])

            return model.fit(
                x, y,
                batch_size=batch_size,
                validation_data=validation_data,
                **kwargs
            )

    tuner = kt.RandomSearch(
        LSTMHyperModel(input_shape=(look_back, 1)),
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

    data_info = {
        'raw_data': data,
        'normalized_data': normalized_data,
        'scaling_info': (min_val, max_val),
        'ticker': ticker
    }

    return tuner, data_info


def prepare_data_with_lookback(normalized_data, look_back):
    """
    Prepare dataset with a specific look_back period.

    Args:
        normalized_data (array): Normalized input data.
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) prepared data arrays.
    """
    X, y = create_dataset(normalized_data, look_back)

    X_train, y_train, X_test, y_test = split_train_test(X, y, TEST_SIZE)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return X_train, y_train, X_test, y_test


def train_best_model(tuner, data_info, train_look_back, train_epochs, batch_fraction=0.01, verbose=True):
    """
    Train the best model found by the tuner with a potentially different look_back and epochs.

    Args:
        tuner (RandomSearch): Keras Tuner object with best hyperparameters.
        data_info (dict): Dictionary containing data-related information.
        train_look_back (int): Number of previous time steps to use as input features for training.
        train_epochs (int): Number of training epochs.
        batch_fraction (float, optional): Fraction of training data to use for batch size calculation.
            Only used as fallback if batch size not in hyperparameters. Defaults to 0.01.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Results dictionary containing model, predictions, dates, metrics, and configuration details.
    """
    normalized_data = data_info['normalized_data']
    min_val, max_val = data_info['scaling_info']
    ticker = data_info['ticker']

    X_train, y_train, X_test, y_test = prepare_data_with_lookback(normalized_data, train_look_back)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = build_lstm_model(best_hp, input_shape=(train_look_back, 1))

    if 'batch_size' in best_hp.values:
        batch_size = best_hp.values['batch_size']
        if verbose:
            print(f"Using tuned batch size: {batch_size}")
    else:
        batch_size = int(len(X_train) * batch_fraction)
        batch_size = max(1, batch_size)
        if verbose:
            print(f"Using calculated batch size: {batch_size}")

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=train_epochs, verbose=verbose)

    train_preds = model.predict(X_train, verbose=verbose)
    test_preds = model.predict(X_test, verbose=verbose)

    rmse_normalized = tf.sqrt(tf.reduce_mean(tf.square(test_preds - y_test))).numpy()

    train_preds_original = inverse_scale(train_preds, min_val, max_val)
    test_preds_original = inverse_scale(test_preds, min_val, max_val)
    y_train_original = inverse_scale(y_train, min_val, max_val)
    y_test_original = inverse_scale(y_test, min_val, max_val)

    rmse_original = tf.sqrt(tf.reduce_mean(tf.square(test_preds_original - y_test_original))).numpy()

    if verbose:
        print("Best Hyperparameters:", best_hp.values)
        print(f"Look-back period: {train_look_back}")
        print(f"Training epochs: {train_epochs}")
        print(f"Normalized RMSE: {rmse_normalized}")
        print(f"Original RMSE: {rmse_original}")

    raw_data = data_info['raw_data']
    data_dates = raw_data.index

    train_dates = data_dates[:len(y_train_original)]
    test_dates = data_dates[-len(y_test_original):]

    results = {
        'model': model,
        'predictions': (train_preds_original, test_preds_original, y_train_original, y_test_original),
        'dates': (train_dates, test_dates),
        'metrics': {'normalized_rmse': rmse_normalized, 'original_rmse': rmse_original},
        'hyperparameters': best_hp.values
    }

    results['config'] = {
        'look_back': train_look_back,
        'epochs': train_epochs,
    }

    if 'history' in locals():
        results['history'] = history.history

    return results