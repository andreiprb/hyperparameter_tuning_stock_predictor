import numpy as np
import tensorflow as tf
import random
import os
import sys

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tuning import tune_hyperparameters, train_best_model
from simple_training import run_model
from utility import inverse_scale, plot_results, prepare_stock_data
from constants import RANDOM_SEED, LOOK_BACK, TUNING_EPOCHS, TRAINING_EPOCHS, TUNING_MAX_TRIALS, \
    TUNING_EXECUTIONS_PER_TRIAL, TEST_SIZE, BATCH_FRACTION, TRAIN_TICKERS_COUNT, START_DATE, END_DATE, \
    TUNING_DATA_FRACTION, TICKERS


def setup_logging():
    """Set up logging to a file in the results folder."""
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, "_results.txt")


class StdoutRedirector:
    """Class to redirect stdout and filter out progress bars."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
        self.is_progress_bar = False

    def write(self, message):
        self.terminal.write(message)
        if '━' in message:
            self.is_progress_bar = True
            return
        if self.is_progress_bar and ('/' in message or message.strip().isdigit()):
            return
        if self.is_progress_bar and message.strip() == '':
            self.is_progress_bar = False
        if not self.is_progress_bar:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def prepare_combined_data(tickers, start_date, end_date, look_back, verbose=True):
    """Prepare combined dataset from multiple tickers using existing utilities."""
    if verbose:
        print(f"Downloading and preparing data for {len(tickers)} tickers...")

    all_data = []
    ticker_indices = {}
    ticker_scaling_info = {}
    current_index = 0

    for ticker in tickers:
        if verbose:
            print(f"Processing {ticker}...")

        prepared_data = prepare_stock_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            look_back=look_back,
            verbose=False
        )

        if prepared_data is None:
            if verbose:
                print(f"Skipping {ticker} due to insufficient data")
            continue

        data_splits = prepared_data['data_splits']
        X_train, y_train, X_test, y_test = data_splits
        min_val, max_val = prepared_data['scaling_info']

        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])

        ticker_indices[ticker] = (current_index, current_index + len(X_combined))
        ticker_scaling_info[ticker] = (min_val, max_val)

        all_data.append((X_combined, y_combined))
        current_index += len(X_combined)

    return {
        'data': all_data,
        'ticker_indices': ticker_indices,
        'ticker_scaling_info': ticker_scaling_info
    }


def combine_datasets(data_list):
    """Combine multiple datasets into a single large dataset."""
    X_combined = np.vstack([X for X, _ in data_list])
    y_combined = np.concatenate([y for _, y in data_list])
    return X_combined, y_combined


def train_base_model(train_tickers, verbose=True):
    """
    Train a base model on data from multiple tickers, using a subset for hyperparameter tuning.
    Reuses tuning logic from tuning.py.
    """
    combined_data_info = prepare_combined_data(
        tickers=train_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        look_back=LOOK_BACK,
        verbose=verbose
    )

    X_combined, y_combined = combine_datasets(combined_data_info['data'])

    if verbose:
        print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")

    X_tune, X_remaining, y_tune, y_remaining = train_test_split(
        X_combined, y_combined,
        train_size=TUNING_DATA_FRACTION,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    if verbose:
        print(f"Using {TUNING_DATA_FRACTION * 100:.1f}% of combined data for hyperparameter tuning")
        print(f"Tuning dataset shape: X={X_tune.shape}, y={y_tune.shape}")

    X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
        X_tune, y_tune, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_SEED
    )

    prepared_data = {
        'data_splits': (X_tune_train, y_tune_train, X_tune_val, y_tune_val),
        'scaling_info': (0.0, 1.0)
    }

    tuner, _ = tune_hyperparameters(
        max_trials=TUNING_MAX_TRIALS,
        executions_per_trial=TUNING_EXECUTIONS_PER_TRIAL,
        epochs=TUNING_EPOCHS,
        verbose=verbose,
        prepared_data=prepared_data
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_SEED
    )

    data_splits = (X_train, y_train, X_test, y_test)
    scaling_info = (0.0, 1.0)

    rmse_normalized, rmse_original, model, predictions = train_best_model(
        tuner=tuner,
        data_splits=data_splits,
        scaling_info=scaling_info,
        look_back=LOOK_BACK,
        epochs=TRAINING_EPOCHS,
        verbose=verbose
    )

    combined_data_info['best_hyperparameters'] = tuner.get_best_hyperparameters(num_trials=1)[0].values
    combined_data_info['metrics'] = {
        'normalized_rmse': rmse_normalized,
        'original_rmse': rmse_original
    }

    return model, combined_data_info

def evaluate_on_test_tickers(model, test_tickers, verbose=True):
    """
    Evaluate the trained model on test tickers not used during training.
    """
    results = {}

    for ticker in test_tickers:
        if verbose:
            print(f"\nEvaluating on {ticker}...")

        prepared_data = prepare_stock_data(
            ticker=ticker,
            start_date=START_DATE,
            end_date=END_DATE,
            look_back=LOOK_BACK,
            test_ratio=TEST_SIZE,
            verbose=verbose
        )

        if prepared_data is None:
            if verbose:
                print(f"Skipping {ticker} due to insufficient data")
            continue

        X_train, y_train, X_test, y_test = prepared_data['data_splits']
        min_val, max_val = prepared_data['scaling_info']
        data_dates = prepared_data['dates']

        train_batch_size = max(1, int(len(X_train) * BATCH_FRACTION))
        test_batch_size = max(1, int(len(X_test) * BATCH_FRACTION))

        if verbose:
            print(f"Using prediction batch sizes: train={train_batch_size}, test={test_batch_size}")

        train_preds = model.predict(X_train, batch_size=train_batch_size, verbose=False)
        test_preds = model.predict(X_test, batch_size=test_batch_size, verbose=False)

        if hasattr(test_preds, 'numpy'):
            test_preds = test_preds.numpy()
        if hasattr(y_test, 'numpy'):
            y_test = y_test.numpy()

        rmse_normalized = np.sqrt(mean_squared_error(y_test, test_preds))

        train_preds_original = inverse_scale(train_preds, min_val, max_val)
        test_preds_original = inverse_scale(test_preds, min_val, max_val)
        y_train_original = inverse_scale(y_train, min_val, max_val)
        y_test_original = inverse_scale(y_test, min_val, max_val)

        if hasattr(test_preds_original, 'numpy'):
            test_preds_original = test_preds_original.numpy()
        if hasattr(y_test_original, 'numpy'):
            y_test_original = y_test_original.numpy()

        rmse_original = np.sqrt(mean_squared_error(y_test_original, test_preds_original))

        if verbose:
            print(f"Normalized RMSE: {rmse_normalized:.4f}")
            print(f"Original RMSE: {rmse_original:.4f}")

        train_indices = range(LOOK_BACK, LOOK_BACK + len(y_train))
        test_indices = range(LOOK_BACK + len(y_train), LOOK_BACK + len(y_train) + len(y_test))

        train_dates = data_dates[train_indices]
        test_dates = data_dates[test_indices]

        results[ticker] = {
            'predictions': (train_preds_original, test_preds_original, y_train_original, y_test_original),
            'dates': (train_dates, test_dates),
            'metrics': {'normalized_rmse': rmse_normalized, 'original_rmse': rmse_original}
        }

        if verbose:
            plot_results(ticker, results[ticker], save=True, transfer_learning=True)

    return results


def compare_with_individual_models(test_tickers, test_results, verbose=True):
    """
    Compare transfer learning results with individual models.
    """
    comparison = {}

    for ticker in test_tickers:
        if ticker not in test_results:
            continue

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Training individual model for {ticker}")
            print(f"{'=' * 50}")

        individual_result = run_model(
            ticker=ticker,
            verbose=verbose,
            save_plot=True
        )

        transfer_result = test_results[ticker]

        transfer_rmse = transfer_result['metrics']['original_rmse']
        individual_rmse = individual_result['metrics']['original_rmse']
        improvement = (individual_rmse - transfer_rmse) / individual_rmse * 100

        transfer_norm_rmse = transfer_result['metrics']['normalized_rmse']
        individual_norm_rmse = individual_result['metrics']['normalized_rmse']
        norm_improvement = (individual_norm_rmse - transfer_norm_rmse) / individual_norm_rmse * 100

        comparison[ticker] = {
            'transfer_rmse': transfer_rmse,
            'individual_rmse': individual_rmse,
            'improvement': improvement,
            'transfer_norm_rmse': transfer_norm_rmse,
            'individual_norm_rmse': individual_norm_rmse,
            'norm_improvement': norm_improvement
        }

        if verbose:
            print(f"\nComparison for {ticker}:")
            print(f"Transfer Learning: Normalized RMSE: {transfer_norm_rmse:.4f}, Original RMSE: {transfer_rmse:.4f}")
            print(
                f"Individual Model: Normalized RMSE: {individual_norm_rmse:.4f}, Original RMSE: {individual_rmse:.4f}")
            print(f"Improvement (Normalized): {norm_improvement:.2f}%")
            print(f"Improvement (Original): {improvement:.2f}%")

    if verbose and comparison:
        print_comparison_summary(comparison)

    return comparison


def print_comparison_summary(comparison):
    """Print summary statistics for the comparison results."""
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    avg_orig_improvement = sum(comp['improvement'] for comp in comparison.values()) / len(comparison)
    avg_norm_improvement = sum(comp['norm_improvement'] for comp in comparison.values()) / len(comparison)
    better_count = sum(1 for comp in comparison.values() if comp['improvement'] > 0)

    print(f"\nTransfer learning was better in {better_count}/{len(comparison)} cases")
    print(f"Average improvement (Normalized RMSE): {avg_norm_improvement:.2f}%")
    print(f"Average improvement (Original RMSE): {avg_orig_improvement:.2f}%")


def run_transfer_learning():
    """
    Run the complete transfer learning process.
    """
    print(f"Starting transfer learning experiment with {len(TICKERS)} tickers")
    print(f"Using batch fraction: {BATCH_FRACTION}")

    random.seed(RANDOM_SEED)
    shuffled_tickers = TICKERS.copy()
    random.shuffle(shuffled_tickers)

    train_tickers = shuffled_tickers[:TRAIN_TICKERS_COUNT]
    test_tickers = shuffled_tickers[TRAIN_TICKERS_COUNT:]

    print(f"Selected {len(train_tickers)} tickers for training (90% of all tickers)")
    print(f"Selected {len(test_tickers)} tickers for testing (10% of all tickers)")
    print(f"Train tickers: {train_tickers}")
    print(f"Test tickers: {test_tickers}")

    print("Training base model on combined data from training tickers...")
    base_model, combined_data_info = train_base_model(
        train_tickers=train_tickers,
        verbose=True,
    )

    print("\nEvaluating on test tickers...")
    test_results = evaluate_on_test_tickers(
        model=base_model,
        test_tickers=test_tickers,
        verbose=True,
    )

    return base_model, train_tickers, test_tickers, test_results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    log_file = setup_logging()

    original_stdout = sys.stdout
    sys.stdout = StdoutRedirector(log_file)

    try:
        base_model, train_tickers, test_tickers, test_results = run_transfer_learning()
        compare_with_individual_models(test_tickers, test_results)
    finally:
        if isinstance(sys.stdout, StdoutRedirector):
            sys.stdout.close()
        sys.stdout = original_stdout