import yfinance as yf
from .data_utils import create_sequences
import pandas as pd

def download_data(ticker, start_date, end_date, interval):
    """
    Downloads stock data for a given ticker and time period using yfinance.
    """
    # Download the data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Drop any rows with missing data
    data = data.dropna()

    # Convert the data to a list of dictionaries
    data = data.reset_index().to_dict('records')

    return data

def get_dataset(ticker, start_date, end_date, interval, sequence_length):
    """
    Downloads stock data for a given ticker and time period, and preprocesses it into sequences of a given length.
    """
    # Download the data
    data = download_data(ticker, start_date, end_date, interval)

    # Extract the closing prices from the data
    prices = [row['Close'] for row in data]

    # Create sequences of the given length from the data
    sequences = create_sequences(prices, sequence_length)

    # Create a Pandas DataFrame from the sequences
    dataset = pd.DataFrame(sequences, columns=['Sequence', 'Label'])

    return dataset


def load_stock_data(symbol, start_date, end_date):
    """
    Loads historical stock price data for a given symbol and date range.

    Args:
        symbol (str): Ticker symbol for the stock.
        start_date (str): Start date in the format "YYYY-MM-DD".
        end_date (str): End date in the format "YYYY-MM-DD".

    Returns:
        pandas.DataFrame: A DataFrame containing the historical stock price data.
    """
    # Define the URL for the API call
    url = f"https://api.polygon.io/v1/historic/agg/day/{symbol}/range/1/day/{start_date}/{end_date}"

    # Make the API call and convert the response to a DataFrame
    response = pd.read_json(url)
    data = pd.DataFrame(response["results"])

    # Filter the DataFrame to only include the relevant columns
    data = data[["open", "high", "low", "close", "volume", "timestamp"]]

    # Rename the columns to be more descriptive
    data.columns = ["Open", "High", "Low", "Close", "Volume", "Date"]

    # Convert the Date column to a pandas datetime object
    data["Date"] = pd.to_datetime(data["Date"], unit="ms").dt.date

    # Set the Date column as the DataFrame index
    data.set_index("Date", inplace=True)

    return data