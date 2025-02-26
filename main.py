import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tuning import tune_hyperparameters, train_best_model

RANDOM_SEED = 42
DEFAULT_BATCH_FRACTION = 0.01

TUNING_LOOK_BACK = 60
TUNING_MAX_TRIALS = 20
TUNING_EXECUTIONS_PER_TRIAL = 1
TUNING_EPOCHS = 10

TRAINING_CONFIGS = [
    {"look_back": 30, "epochs": 15},
    {"look_back": 60, "epochs": 20},
    {"look_back": 90, "epochs": 25},
]

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


def run_model(ticker, training_configs=None, verbose=True):
    """
    Run the complete model process for a given ticker.

    The process includes:
    1. Tuning hyperparameters
    2. Training the best model with different look_back and epochs configurations

    Args:
        ticker (str): Stock ticker symbol.
        training_configs (list, optional): List of dictionaries containing look_back and epochs values.
            Defaults to TRAINING_CONFIGS.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Results of the best model configuration including predictions, metrics, and configuration details.
    """
    if training_configs is None:
        training_configs = TRAINING_CONFIGS

    if verbose:
        print(f"Tuning hyperparameters for {ticker}...")

    tuner, data_info = tune_hyperparameters(
        ticker=ticker,
        look_back=TUNING_LOOK_BACK,
        max_trials=TUNING_MAX_TRIALS,
        executions_per_trial=TUNING_EXECUTIONS_PER_TRIAL,
        epochs=TUNING_EPOCHS,
        verbose=verbose
    )

    best_result = None
    best_rmse = float('inf')

    for i, config in enumerate(training_configs):
        if verbose:
            print(f"\nTraining configuration {i + 1}/{len(training_configs)}:")
            print(f"Look-back: {config['look_back']}, Epochs: {config['epochs']}")

        result = train_best_model(
            tuner=tuner,
            data_info=data_info,
            train_look_back=config['look_back'],
            train_epochs=config['epochs'],
            batch_fraction=DEFAULT_BATCH_FRACTION,
            verbose=verbose
        )

        if result['metrics']['original_rmse'] < best_rmse:
            best_rmse = result['metrics']['original_rmse']
            best_result = result

    if verbose:
        best_config = best_result['config']

        print(f"\nBest configuration for {ticker}:")
        print(f"Look-back: {best_config['look_back']}")
        print(f"Epochs: {best_config['epochs']}")
        print(f"Normalized RMSE: {best_result['metrics']['normalized_rmse']:.4f}")
        print(f"Original RMSE: {best_rmse:.4f}")

        plot_results(ticker, best_result)

    return best_result


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