import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import yfinance as yf
import keras_tuner as kt
import os
import sys
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tuning import build_lstm_model
from utility import min_max_scale, inverse_scale, create_dataset
from simple_training import TICKERS, RANDOM_SEED, LOOK_BACK, TRAINING_EPOCHS, TUNING_EPOCHS, run_model

# Constants
TRAIN_TICKERS_COUNT = 45  # Use 45 stocks for training (90% of 50)
TEST_SIZE = 0.05
TUNING_MAX_TRIALS = 15
TUNING_EXECUTIONS_PER_TRIAL = 1
BATCH_SIZE = 32  # Fixed batch size, not tuned
START_DATE = "2020-01-01"
END_DATE = "2023-01-01"

# Subset for hyperparameter tuning
TUNING_DATA_FRACTION = 0.3  # Use 30% of data for hyperparameter tuning


# Set up logging to capture all output
def setup_logging():
    """Set up logging to a file in the results folder."""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Create log file
    log_file = os.path.join(results_dir, "_results.txt")

    return log_file


class StdoutRedirector:
    """Class to redirect stdout and filter out progress bars."""

    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')
        self.is_progress_bar = False

    def write(self, message):
        self.terminal.write(message)
        # Skip progress bar lines containing ━
        if '━' in message:
            self.is_progress_bar = True
            return
        # Skip lines with numeric/ patterns from tqdm
        if self.is_progress_bar and ('/' in message or message.strip().isdigit()):
            return
        # Reset progress bar flag when we hit a newline
        if self.is_progress_bar and message.strip() == '':
            self.is_progress_bar = False
        # Write to log if not a progress bar
        if not self.is_progress_bar:
            self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def prepare_combined_data(tickers, start_date, end_date, look_back, verbose=True):
    """
    Prepare combined dataset from multiple tickers.

    Args:
        tickers (list): List of ticker symbols to download and combine.
        start_date (str): Start date for data download.
        end_date (str): End date for data download.
        look_back (int): Number of previous time steps to use as input features.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Dictionary containing combined data, scaling information, and ticker mapping.
    """
    if verbose:
        print(f"Downloading and preparing data for {len(tickers)} tickers...")

    all_normalized_data = []
    ticker_indices = {}
    ticker_scaling_info = {}
    current_index = 0

    for ticker in tickers:
        if verbose:
            print(f"Processing {ticker}...")

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty or len(data) < look_back + 10:
            if verbose:
                print(f"Skipping {ticker} due to insufficient data")
            continue

        close_data = data[['Close']].dropna().values

        normalized_data, min_val, max_val = min_max_scale(close_data)
        X, y = create_dataset(normalized_data, look_back)

        ticker_indices[ticker] = (current_index, current_index + len(X))
        ticker_scaling_info[ticker] = (min_val, max_val)

        all_normalized_data.append((X.reshape(X.shape[0], X.shape[1], 1), y))
        current_index += len(X)

    return {
        'data': all_normalized_data,
        'ticker_indices': ticker_indices,
        'ticker_scaling_info': ticker_scaling_info
    }


def combine_datasets(data_list):
    """
    Combine multiple datasets into a single large dataset.

    Args:
        data_list (list): List of (X, y) tuples to combine.

    Returns:
        tuple: (X_combined, y_combined) combined data arrays.
    """
    X_combined = np.vstack([X for X, _ in data_list])
    y_combined = np.concatenate([y for _, y in data_list])

    return X_combined, y_combined


def train_base_model(train_tickers, verbose=True):
    """
    Train a base model on data from multiple tickers, using a subset for hyperparameter tuning.

    Args:
        train_tickers (list): List of ticker symbols to use for training.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        tuple: (model, combined_data_info) where model is the trained model and
               combined_data_info contains information about the combined dataset.
    """
    # Prepare combined data for all training tickers
    combined_data_info = prepare_combined_data(
        tickers=train_tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        look_back=LOOK_BACK,  # Use fixed LOOK_BACK from simple_training
        verbose=verbose
    )

    # Combine all datasets
    X_combined, y_combined = combine_datasets(combined_data_info['data'])

    if verbose:
        print(f"Combined dataset shape: X={X_combined.shape}, y={y_combined.shape}")

    # For hyperparameter tuning, use only a subset of the data
    X_tune, _, y_tune, _ = train_test_split(
        X_combined, y_combined,
        train_size=TUNING_DATA_FRACTION,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    if verbose:
        print(f"Using {TUNING_DATA_FRACTION * 100:.1f}% of combined data for hyperparameter tuning")
        print(f"Tuning dataset shape: X={X_tune.shape}, y={y_tune.shape}")

    # Split tuning data into training and validation sets
    X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
        X_tune, y_tune, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_SEED
    )

    if verbose:
        print("Tuning hyperparameters on subset of combined dataset...")

    # Create a tuner for the combined dataset
    input_shape = (LOOK_BACK, 1)

    tuner = kt.RandomSearch(
        lambda hp: build_lstm_model(hp, input_shape),
        objective='val_loss',
        max_trials=TUNING_MAX_TRIALS,
        executions_per_trial=TUNING_EXECUTIONS_PER_TRIAL,
        directory='hyper_tuning',
        project_name='_combined_lstm_tuning'
    )

    # Search for best hyperparameters on the subset
    tuner.search(
        X_tune_train, y_tune_train,
        validation_data=(X_tune_val, y_tune_val),
        epochs=TUNING_EPOCHS,
        batch_size=BATCH_SIZE,  # Use fixed batch size
        verbose=verbose
    )

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    if verbose:
        print("\nBest Hyperparameters:")
        for param, value in best_hp.values.items():
            print(f"{param}: {value}")

    # Build the model with best hyperparameters
    model = build_lstm_model(best_hp, input_shape)

    # Train the model on the entire combined dataset with the best hyperparameters
    if verbose:
        print("\nTraining final model on the FULL combined dataset...")

    history = model.fit(
        X_combined, y_combined,
        batch_size=BATCH_SIZE,  # Use fixed batch size
        epochs=TRAINING_EPOCHS,  # Use TRAINING_EPOCHS from simple_training
        validation_split=TEST_SIZE,
        verbose=verbose
    )

    combined_data_info['best_hyperparameters'] = best_hp.values
    combined_data_info['history'] = history.history

    return model, combined_data_info


def plot_transfer_results(ticker, results, save=False):
    """
    Plot the results of model evaluation for transfer learning.

    Args:
        ticker (str): Stock ticker symbol.
        results (dict): Dictionary containing model predictions, dates, and metrics.
        save (bool): Whether to save the plot to a file.
    """
    train_preds_original, test_preds_original, y_train_original, y_test_original = results['predictions']
    train_dates, test_dates = results['dates']
    metrics = results['metrics']

    plt.figure(figsize=(16, 8))
    plt.title(
        f'Transfer Learning LSTM Model - {ticker}\nNormalized RMSE: {metrics["normalized_rmse"]:.4f}, Original RMSE: {metrics["original_rmse"]:.4f}')
    plt.xlabel('Date')
    plt.ylabel('Closing Price USD ($)')

    plt.plot(train_dates, y_train_original, color='blue', label='Real - Train')
    plt.plot(train_dates, train_preds_original, color='orange', label='Predicted - Train')

    plt.plot(test_dates, y_test_original, color='green', label='Real - Test')
    plt.plot(test_dates, test_preds_original, color='red', label='Predicted - Test')

    plt.legend()
    plt.grid(True, alpha=0.3)

    if save:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)

        # Save the plot
        plt.savefig(os.path.join(results_dir, f"{ticker}_transfer_learning.png"))

    plt.show()


def evaluate_on_test_tickers(model, test_tickers, verbose=True):
    """
    Evaluate the trained model on test tickers not used during training.

    Args:
        model (Model): Trained Keras model.
        test_tickers (list): List of ticker symbols to use for testing.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Dictionary mapping each ticker to its evaluation metrics.
    """
    results = {}

    for ticker in test_tickers:
        if verbose:
            print(f"\nEvaluating on {ticker}...")

        # Download data for the ticker
        data = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            progress=False
        )

        if data.empty or len(data) < LOOK_BACK + 10:
            if verbose:
                print(f"Skipping {ticker} due to insufficient data")
            continue

        close_data = data[['Close']].dropna().values

        # Normalize the data
        normalized_data, min_val, max_val = min_max_scale(close_data)

        # Create sequences
        X, y = create_dataset(normalized_data, LOOK_BACK)  # Use fixed LOOK_BACK
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into train and test sets
        train_size = int(len(X) * (1 - TEST_SIZE))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Generate predictions
        train_preds = model.predict(X_train, verbose=False)
        test_preds = model.predict(X_test, verbose=False)

        # Convert TensorFlow tensors to NumPy arrays if needed
        if hasattr(test_preds, 'numpy'):
            test_preds = test_preds.numpy()
        if hasattr(y_test, 'numpy'):
            y_test = y_test.numpy()

        # Calculate normalized RMSE
        rmse_normalized = np.sqrt(mean_squared_error(y_test, test_preds))

        # Convert back to original scale
        train_preds_original = inverse_scale(train_preds, min_val, max_val)
        test_preds_original = inverse_scale(test_preds, min_val, max_val)
        y_train_original = inverse_scale(y_train, min_val, max_val)
        y_test_original = inverse_scale(y_test, min_val, max_val)

        # Convert to NumPy arrays for RMSE calculation
        if hasattr(test_preds_original, 'numpy'):
            test_preds_original = test_preds_original.numpy()
        if hasattr(y_test_original, 'numpy'):
            y_test_original = y_test_original.numpy()

        # Calculate original RMSE
        rmse_original = np.sqrt(mean_squared_error(y_test_original, test_preds_original))

        if verbose:
            print(f"Normalized RMSE: {rmse_normalized:.4f}")
            print(f"Original RMSE: {rmse_original:.4f}")
            print(f"(Lower RMSE values indicate better predictions)")

        # Prepare dates for plotting
        data_dates = data.index
        train_dates = data_dates[LOOK_BACK:LOOK_BACK + len(y_train)]
        test_dates = data_dates[LOOK_BACK + len(y_train):LOOK_BACK + len(y_train) + len(y_test)]

        # Store results
        results[ticker] = {
            'predictions': (train_preds_original, test_preds_original, y_train_original, y_test_original),
            'dates': (train_dates, test_dates),
            'metrics': {'normalized_rmse': rmse_normalized, 'original_rmse': rmse_original}
        }

        # Plot results and save
        if verbose:
            plot_transfer_results(ticker, results[ticker], save=True)

    return results


def compare_with_individual_models(test_tickers, test_results, verbose=True):
    """
    Compare transfer learning results with individual models trained on each test ticker.

    Args:
        test_tickers (list): List of ticker symbols used for testing.
        test_results (dict): Results from transfer learning evaluation.
        verbose (bool, optional): Whether to print progress information. Defaults to True.

    Returns:
        dict: Comparison results.
    """
    comparison = {}

    for ticker in test_tickers:
        if ticker not in test_results:
            continue

        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Training individual model for {ticker}")
            print(f"{'=' * 50}")

        # Train individual model
        individual_result = run_model(
            ticker=ticker,
            verbose=verbose
        )

        # Get transfer learning result
        transfer_result = test_results[ticker]

        # Extract predictions and metrics
        transfer_rmse = transfer_result['metrics']['original_rmse']
        individual_rmse = individual_result['metrics']['original_rmse']
        improvement = (individual_rmse - transfer_rmse) / individual_rmse * 100

        # Extract normalized RMSEs
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

        # Save the individual model's plot as well
        try:
            # Get the individual model plot data
            train_preds_original, test_preds_original, y_train_original, y_test_original = individual_result['predictions']
            train_dates, test_dates = individual_result['dates']

            # Create a new figure
            plt.figure(figsize=(16, 8))
            plt.title(
                f'Individual LSTM Model - {ticker}\nNormalized RMSE: {individual_norm_rmse:.4f}, Original RMSE: {individual_rmse:.4f}')
            plt.xlabel('Date')
            plt.ylabel('Closing Price USD ($)')

            plt.plot(train_dates, y_train_original, color='blue', label='Real - Train')
            plt.plot(train_dates, train_preds_original, color='orange', label='Predicted - Train')

            plt.plot(test_dates, y_test_original, color='green', label='Real - Test')
            plt.plot(test_dates, test_preds_original, color='red', label='Predicted - Test')

            plt.legend()
            plt.grid(True, alpha=0.3)

            # Save the plot
            results_dir = os.path.join(os.getcwd(), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plt.savefig(os.path.join(results_dir, f"{ticker}_individual_model.png"))

            plt.show()
        except Exception as e:
            print(f"Error saving individual model plot for {ticker}: {e}")

    # Summarize comparison
    if verbose and comparison:
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)

        avg_orig_improvement = sum(comp['improvement'] for comp in comparison.values()) / len(comparison)
        avg_norm_improvement = sum(comp['norm_improvement'] for comp in comparison.values()) / len(comparison)
        better_count = sum(1 for comp in comparison.values() if comp['improvement'] > 0)

        print(f"\nTransfer learning was better in {better_count}/{len(comparison)} cases")
        print(f"Average improvement (Normalized RMSE): {avg_norm_improvement:.2f}%")
        print(f"Average improvement (Original RMSE): {avg_orig_improvement:.2f}%")

    return comparison


def run_transfer_learning():
    """
    Run the complete transfer learning process.

    This function:
    1. Uses 30% of data from the 45 training tickers for hyperparameter tuning
    2. Uses 90% of tickers (all data) for model training
    3. Tests on 10% of tickers

    Only model architecture hyperparameters are tuned, not look_back or batch_size.

    Returns:
        tuple: (base_model, train_tickers, test_tickers, test_results)
    """
    print(f"Starting transfer learning experiment with {len(TICKERS)} tickers")

    # Randomly select train and test tickers
    random.seed(RANDOM_SEED)
    shuffled_tickers = TICKERS.copy()
    random.shuffle(shuffled_tickers)

    # Use 90% of tickers for training, 10% for testing
    train_tickers = shuffled_tickers[:TRAIN_TICKERS_COUNT]
    test_tickers = shuffled_tickers[TRAIN_TICKERS_COUNT:]

    print(f"Selected {len(train_tickers)} tickers for training (90% of all tickers)")
    print(f"Selected {len(test_tickers)} tickers for testing (10% of all tickers)")
    print(f"Train tickers: {train_tickers}")
    print(f"Test tickers: {test_tickers}")

    # Train the base model
    print("Training base model on combined data from training tickers...")
    base_model, combined_data_info = train_base_model(train_tickers, verbose=True)

    # Evaluate on test tickers
    print("\nEvaluating on test tickers...")
    test_results = evaluate_on_test_tickers(
        model=base_model,
        test_tickers=test_tickers,
        verbose=True
    )

    return base_model, train_tickers, test_tickers, test_results


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Set up logging
    log_file = setup_logging()

    # Redirect stdout to file and terminal
    original_stdout = sys.stdout
    sys.stdout = StdoutRedirector(log_file)

    try:
        # Run transfer learning
        base_model, train_tickers, test_tickers, test_results = run_transfer_learning()

        # Compare with individual models
        compare_with_individual_models(test_tickers, test_results)

    finally:
        # Restore stdout
        if isinstance(sys.stdout, StdoutRedirector):
            sys.stdout.close()
        sys.stdout = original_stdout