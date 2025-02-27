import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yfinance as yf

from tuning import tune_hyperparameters, train_best_model

RANDOM_SEED = 42
DEFAULT_BATCH_FRACTION = 0.01

LOOK_BACK = 60
TUNING_EPOCHS = 10
TRAINING_EPOCHS = 20

TUNING_MAX_TRIALS = 20
TUNING_EXECUTIONS_PER_TRIAL = 1

TICKERS = ["AAPL", "ABBV", "ADBE", "AMD", "AMZN",
           "BA", "BABA", "BLK", "COST", "CRM",
           "CRWD", "CVX", "DHR", "DIS", "GD",
           "GME", "GOOG", "GOOGL", "GS", "HD",
           "IBM", "INTC", "JPM", "KO", "LMT",
           "MA", "META", "MRNA", "MSFT", "NFLX",
           "NKE", "NOW", "NVDA", "ORCL", "PEP",
           "PFE", "PG", "PYPL", "QCOM", "SHOP",
           "T", "TMO", "TSLA", "TXN", "UNH",
           "V", "VZ", "WMT", "XOM", "ZS"]

DEFAULT_TICKER = "PFE"


def plot_results(ticker, results):
    """
    Plot the results of model training and evaluation.

    Args:
        ticker (str): Stock ticker symbol.
        results (dict): Dictionary containing model predictions, dates, and metrics.
            Expected format:
            {
                'predictions': (train_preds_original, test_preds_original, y_train_original, y_test_original),
                'dates': (train_dates, test_dates),
                'metrics': {'original_rmse': float, ...}
            }

    Returns:
        None: Displays a plot showing real vs. predicted values for training and test data.
    """
    train_preds_original, test_preds_original, y_train_original, y_test_original = results['predictions']
    train_dates, test_dates = results['dates']
    metrics = results['metrics']

    # Ensure shapes match by trimming if necessary
    min_train_len = min(len(train_dates), len(y_train_original))
    min_test_len = min(len(test_dates), len(y_test_original))

    train_dates = train_dates[:min_train_len]
    y_train_original = y_train_original[:min_train_len]
    train_preds_original = train_preds_original[:min_train_len]

    test_dates = test_dates[:min_test_len]
    y_test_original = y_test_original[:min_test_len]
    test_preds_original = test_preds_original[:min_test_len]

    plt.figure(figsize=(16, 8))
    plt.title(f'LSTM Model - {ticker} - RMSE: {metrics["original_rmse"]:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price USD ($)')

    plt.plot(train_dates, y_train_original, color='blue', label='Real - Train')
    plt.plot(train_dates, train_preds_original, color='orange', label='Predicted - Train')

    plt.plot(test_dates, y_test_original, color='green', label='Real - Test')
    plt.plot(test_dates, test_preds_original, color='red', label='Predicted - Test')

    plt.legend()
    plt.show()


def run_model(ticker, verbose=True):
    """
    Run the complete model process for a given ticker.

    The process includes:
    1. Tuning hyperparameters
    2. Training the best model with fixed look_back and epochs configurations

    Args:
        ticker (str): Stock ticker symbol.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Results of the best model configuration including predictions, metrics, and configuration details.
    """

    if verbose:
        print(f"Tuning hyperparameters for {ticker}...")

    tuner, data_info = tune_hyperparameters(
        ticker=ticker,
        look_back=LOOK_BACK,
        max_trials=TUNING_MAX_TRIALS,
        executions_per_trial=TUNING_EXECUTIONS_PER_TRIAL,
        epochs=TUNING_EPOCHS,
        verbose=verbose
    )

    # Use the fixed look_back and training_epochs parameters
    rmse_normalized, rmse_original, model, predictions = train_best_model(
        tuner=tuner,
        data_splits=data_info[0],
        scaling_info=data_info[1],
        look_back=LOOK_BACK,
        epochs=TRAINING_EPOCHS,
        verbose=verbose
    )

    # Get data dates
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01", progress=False)
    data = data[['Close']].dropna()

    # Extract the actual training and test data sizes
    X_train, y_train, X_test, y_test = data_info[0]

    # Get dates that match exactly with the number of predictions
    train_preds_original, test_preds_original, y_train_original, y_test_original = predictions

    # Get indices that match the actual data points used in training/testing
    train_indices = range(LOOK_BACK, LOOK_BACK + len(y_train))
    test_indices = range(LOOK_BACK + len(y_train), LOOK_BACK + len(y_train) + len(y_test))

    # Get the dates corresponding to these indices
    train_dates = data.index[train_indices]
    test_dates = data.index[test_indices]

    # Format final results
    result = {
        'predictions': predictions,
        'dates': (train_dates, test_dates),
        'metrics': {
            'normalized_rmse': rmse_normalized,
            'original_rmse': rmse_original
        },
        'model': model
    }

    if verbose:
        plot_results(ticker, result)

    return result


def run_tests():
    """
    Run model for all tickers in the TICKERS list.

    This function processes each ticker in the TICKERS list, trains a model for each,
    and ranks the tickers based on the model's RMSE performance.

    Returns:
        dict: Dictionary mapping each ticker to its model results.
    """
    results = {}

    for ticker in TICKERS:
        print(f"\n{'=' * 50}")
        print(f"Processing {ticker}")
        print(f"{'=' * 50}")

        result = run_model(ticker, verbose=True)
        results[ticker] = result

    ranked_tickers = sorted(results.keys(), key=lambda x: results[x]['metrics']['original_rmse'])

    print("\nTickers ranked by RMSE (best to worst):")
    for i, ticker in enumerate(ranked_tickers):
        rmse = results[ticker]['metrics']['original_rmse']
        print(f"{i + 1}. {ticker}: {rmse:.4f}")

    return results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    run_model(ticker=DEFAULT_TICKER)