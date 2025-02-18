import matplotlib.pyplot as plt

from tuning import *

RANDOM_SEED = 42
BATCH_FRACTION = 0.01

LOOK_BACK = 60

TICKER = "PFE"

TICKERS = ["AAPL", "ADBE", "AMD", "AMZN", "BA",
           "BLK", "CRWD", "GD", "GME", "GOOG",
           "GOOGL", "IBM", "INTC", "LMT", "META",
           "MSFT", "NVDA", "SHOP", "TSLA", "ZS"]

def train_best_model(tuner, data_splits, scaling_info, look_back, epochs, verbose=True):
    X_train, y_train, X_test, y_test = data_splits
    min_val, max_val = scaling_info

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    model = build_lstm_model(best_hp, input_shape=(look_back, 1))

    batch_size = int(len(X_train) * BATCH_FRACTION)
    batch_size = max(1, batch_size)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    train_preds = model.predict(X_train, verbose=verbose)
    test_preds = model.predict(X_test, verbose=verbose)

    rmse_normalized = tf.sqrt(tf.reduce_mean(tf.square(test_preds - y_test))).numpy()

    train_preds_original = inverse_scale(train_preds, min_val, max_val)
    test_preds_original = inverse_scale(test_preds, min_val, max_val)
    y_train_original = inverse_scale(y_train, min_val, max_val)
    y_test_original = inverse_scale(y_test, min_val, max_val)

    rmse_original = tf.sqrt(tf.reduce_mean(tf.square(test_preds_original - y_test_original))).numpy()

    print("Best Hyperparameters:", best_hp.values)
    print(f"Normalized RMSE: {rmse_normalized}")
    print(f"Original   RMSE: {rmse_original}")

    return model, (train_preds_original, test_preds_original, y_train_original, y_test_original)


def run_model(ticker, verbose=True):
    tuner, data_splits, scaling_info = tune_hyperparameters(
        ticker=ticker,
        look_back=LOOK_BACK,
        max_trials=20,
        executions_per_trial=1,
        epochs=10,
        verbose=verbose
    )

    final_model, preds = train_best_model(
        tuner,
        data_splits,
        scaling_info,
        LOOK_BACK,
        epochs=10,
        verbose=verbose
    )

    train_preds_original, test_preds_original, y_train_original, y_test_original = preds

    data_plot = yf.download(ticker, start="2020-01-01", end="2023-01-01", progress=False)
    data_plot = data_plot[['Close']].dropna()

    train_dates = data_plot.index[:len(y_train_original)]
    test_dates = data_plot.index[-len(y_test_original):]

    plt.figure(figsize=(16, 8))
    plt.title(f'LSTM Model - Hyperparameter Tuning - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price USD ($)')

    plt.plot(train_dates, y_train_original, color='blue', label='Real - Train')
    plt.plot(train_dates, train_preds_original, color='orange', label='Predicted - Train')

    plt.plot(test_dates, y_test_original, color='green', label='Real - Test')
    plt.plot(test_dates, test_preds_original, color='red', label='Predicted - Test')

    plt.legend()
    plt.show()


def run_tests():
    for ticker in TICKERS:
        run_model(ticker, verbose=False)


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    run_model(ticker=TICKER)