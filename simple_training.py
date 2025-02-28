import numpy as np
import tensorflow as tf
import os

from tuning import tune_hyperparameters, train_best_model
from utility import plot_results, prepare_stock_data
from constants import RANDOM_SEED, LOOK_BACK, TUNING_EPOCHS, TRAINING_EPOCHS, TUNING_MAX_TRIALS, TUNING_EXECUTIONS_PER_TRIAL, TICKERS, DEFAULT_TICKER


def run_model(ticker, verbose=True, save_plot=False):
    """
    Run the complete model process for a given ticker.

    The process includes:
    1. Tuning hyperparameters
    2. Training the best model with fixed look_back and epochs configurations

    Args:
        ticker (str): Stock ticker symbol.
        verbose (bool, optional): Whether to print progress information. Defaults to True.
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to False.

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

    rmse_normalized, rmse_original, model, predictions = train_best_model(
        tuner=tuner,
        data_splits=data_info[0],
        scaling_info=data_info[1],
        look_back=LOOK_BACK,
        epochs=TRAINING_EPOCHS,
        verbose=verbose
    )

    X_train, y_train, X_test, y_test = data_info[0]

    prepared_data = prepare_stock_data(
        ticker=ticker,
        start_date="2020-01-01",
        end_date="2023-01-01",
        look_back=LOOK_BACK,
        verbose=False
    )

    data_dates = prepared_data['dates']

    train_indices = range(LOOK_BACK, LOOK_BACK + len(y_train))
    test_indices = range(LOOK_BACK + len(y_train), LOOK_BACK + len(y_train) + len(y_test))

    train_dates = data_dates[train_indices]
    test_dates = data_dates[test_indices]

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
        plot_results(ticker, result, save=save_plot)

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

    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for ticker in TICKERS:
        print(f"\n{'=' * 50}")
        print(f"Processing {ticker}")
        print(f"{'=' * 50}")

        result = run_model(ticker, verbose=True, save_plot=True)
        results[ticker] = result

    ranked_tickers = sorted(results.keys(), key=lambda x: results[x]['metrics']['original_rmse'])

    print("\nTickers ranked by RMSE (best to worst):")
    for i, ticker in enumerate(ranked_tickers):
        rmse = results[ticker]['metrics']['original_rmse']
        normalized_rmse = results[ticker]['metrics']['normalized_rmse']
        print(f"{i + 1}. {ticker}: Original RMSE={rmse:.4f}, Normalized RMSE={normalized_rmse:.4f}")

    return results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    run_model(ticker=DEFAULT_TICKER, save_plot=True)