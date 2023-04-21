import pandas as pd
import numpy as np
from data import load_stock_data
from data_utils import create_sequences, split_train_test_data


class DataManager:
    """
    Class for managing the downloading and preprocessing of stock data for multiple stocks.
    """
    def __init__(self, stocks, start_date, end_date, sequence_length, train_test_split):
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.train_test_split = train_test_split

    def load_data(self):
        """
        Loads the stock data for the given stocks and date range.
        """
        data = {}
        for stock in self.stocks:
            df = load_stock_data(stock, self.start_date, self.end_date)
            data[stock] = df

        return data

    def preprocess_data(self, data):
        """
        Preprocesses the stock data by creating sequences and splitting into training and testing sets.
        """
        X_train, y_train, X_test, y_test = {}, {}, {}, {}
        for stock in self.stocks:
            df = data[stock]
            df = df.dropna()

            # Create sequences
            sequences = create_sequences(df['Close'].tolist(), self.sequence_length)

            # Split data into training and testing sets
            X_train[stock], y_train[stock], X_test[stock], y_test[stock] = split_train_test_data(sequences, self.train_test_split)

        return X_train, y_train, X_test, y_test

    def download_preprocess_data(self):
        """
        Downloads and preprocesses the stock data.
        """
        data = self.load_data()
        X_train, y_train, X_test, y_test = self.preprocess_data(data)

        return X_train, y_train, X_test, y_test