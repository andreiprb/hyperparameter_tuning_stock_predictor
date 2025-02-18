import yfinance as yf

from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Input, Dropout
from keras.api.optimizers import Adam

import keras_tuner as kt

from utility import *

TEST_SIZE = 0.05

def build_lstm_model(hp, input_shape):
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

    return tuner, (X_train, y_train, X_test, y_test), (min_val, max_val)

