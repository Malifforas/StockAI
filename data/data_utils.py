import os
import pandas as pd
import numpy as np
import yfinance as yf


def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock price data for a given ticker and date range.
    """
    # Download the stock data using Yahoo Finance API
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Drop any rows with missing values
    stock_data = stock_data.dropna()

    # Return the stock data
    return stock_data


def process_stock_data(stock_data, feature_columns=['Close'], target_column='Close', sequence_length=30):
    """
    Processes historical stock price data for use in an AI model.
    """
    # Create a copy of the stock data with only the desired feature columns
    data = stock_data[feature_columns].copy()

    # Normalize the data using the min-max scaling method
    data = (data - data.min()) / (data.max() - data.min())

    # Create sequences of a given length from the normalized data
    X, y = create_sequences(data, sequence_length)

    # Split the data into training and testing sets
    split_index = int(len(X) * 0.8)
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]

    # Return the processed data
    return X_train, y_train, X_test, y_test


def create_sequences(dataset, sequence_length):
    """
    Creates sequences of a given length from a given dataset of stock prices.
    """
    # Convert the dataset to a numpy array
    data = np.array(dataset)

    # Create empty lists to store the sequences and their corresponding labels
    X, y = [], []

    # Loop through the data and create sequences
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])

    # Convert the sequences and labels to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Return the sequences and labels
    return X, y


def save_data(X_train, y_train, X_test, y_test, save_dir='./data/'):
    """
    Saves processed stock price data to disk.
    """
    # Create the save directory if it doesn't already exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the data to numpy binary files
    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)


def load_data(load_dir='./data/'):
    """
    Loads processed stock price data from disk.
    """
    # Load the data from numpy binary files
    X_train = np.load(os.path.join(load_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(load_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(load_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(load_dir, 'y_test.npy'))

    # Return the loaded data
    return X_train, y_train, X_test, y_test
def split_train_test_data(data, train_size=0.8):
    """
    Splits a given dataset into training and testing sets.

    Args:
        data (numpy.ndarray): The dataset to split.
        train_size (float): The proportion of the data to use for training.

    Returns:
        tuple: A tuple containing the training and testing datasets.
    """
    # Calculate the number of rows to use for training
    train_size = int(len(data) * train_size)

    # Split the data into training and testing sets
    train_data = data[:train_size]
    test_data = data[train_size:]

    return train_data, test_data