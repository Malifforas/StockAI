from data.data_manager import DataManager
from models.stock_price_prediction_model import StockPricePredictionModel


# Create a data manager object
data_manager = DataManager()

# Load the stock data
top_500_stocks = data_manager.load_top_500_stocks()
other_stocks = data_manager.load_other_stocks()

# Merge the data from the top 500 stocks and other stocks
merged_data = data_manager.merge_data(top_500_stocks, other_stocks)

# Split the data into training and testing sets
train_data, test_data = data_manager.split_train_test_data(merged_data)

# Create a stock price prediction model
model = StockPricePredictionModel()

# Train the model on the training data
model.train(train_data)

# Test the model on the testing data
predictions = model.test(test_data)

# Print the predictions
print(predictions)