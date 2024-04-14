import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
from pylab import mpl, plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.layers import Dropout


buy_threshold = 0.85  # only buy if prediction confidence is above 60%
sell_threshold = 0.15  # only sell if prediction confidence is below 40%
#close_long_threshold = 0.40
#close_short_threshold = 0.6
columns = []

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(100)

set_seeds()

# matplotlib parameters
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'


# training dataset (1month of 1h candles)
def prepare_training_data(file):
    """
    Retrieves training data from a csv file, edits it and stores in a DataFrame object

    Parameters:
    - file: path to the csv file

    Returns:
    - a DataFrame object with the prepared training data
    """
    data = pd.read_csv(file, index_col=0, parse_dates=True).dropna()
    data.index = pd.to_datetime(data.index, unit='ms')
    data = data.drop(['close_time', 'ignore', 'quote_volume', 'count', 'taker_buy_volume', 
                    'taker_buy_quote_volume', 'open', 'high', 'low'], axis=1)
    
    return data


def calculate_rsi(data, column='close', period=10):
    """
    Calculate the Relative Strength Index (RSI) of a given DataFrame
    
    Parameters:
    - data: DataFrame object containing the price data
    - column: the name of the column in 'data' which contains the price data
    - period: the period over which to calculate RSI
    
    Returns:
    - A Pandas Series representing the RSI values.
    """
    
    # calculate price changes
    delta = data[column].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # calculate the Exponential Moving Average (EMA) of gains and losses
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    # calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Relative Strength Index (RSI)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_lags(data, rsi_lags, cols, column):
    """
    Calculates lagged data of the chosen field of asset's market data and adds it to the DataFrame 'data'
    
    Parameters:
    - data: DataFrame object containing the price data
    - return_lags: number of lagged rsi values
    - cols: list of columns which we use as input parameters for the neural network
    - column: the name of the column in 'data' which contains the rsi data
    """
    for lag in range (1, rsi_lags+1):
        col=f'{column}_lag{lag}'
        data[col] = data[column].shift(lag)
        cols.append(col)

# first testing dataset (1 month of 1h candles)
#test_data = pd.read_csv('SOLUSDT-1h-2024-02.csv', index_col=0, parse_dates=True).dropna()
#test_data.index = pd.to_datetime(test_data.index, unit='ms')
#test_data = test_data.drop(['close_time', 'ignore', 'quote_volume', 'count', 'taker_buy_volume', 
 #                           'taker_buy_quote_volume', 'open', 'high', 'low'], axis=1)

def add_indicators_to_training_data(data, cols, column='close'):
    """
    Uses given data from the training dataset to derive new values/indicators, and adds them to the DataFrame object

    Parameters:
    - data: prepared DataFrame object with initial training data
    - column: the name of the column in data which contains the price data
    """
    data['returns'] = np.log(data[column] / data[column].shift(1)) 
    data['future_close'] = data[column].shift(1)
    data['future_return'] = np.log(data['future_close']/data[column])
    data['direction'] = np.where(data['future_return'] > 0, 1, 0)
    data['momentum'] = data['returns'].rolling(5).mean().shift(1)
    data['volatility'] = data['returns'].rolling(20).std().shift(1)
    data['distance'] = (data[column] - data[column].rolling(50).mean()).shift(1)
    data['rsi'] = calculate_rsi(data, 'close', 14)
    calculate_lags(data, 10, cols, 'returns')
    calculate_lags(data, 10, cols, 'volume')
    calculate_lags(data, 10, cols, 'rsi')
    #calculate_lags(data, 10, cols, 'momentum')
    #calculate_lags(data, 10, cols, 'distance')
    #calculate_lags(data, 10, cols, 'volatility')
    data.dropna(inplace=True)
    cols.extend(['momentum', 'volatility', 'distance', 'volume', 'rsi'])
    #filtered_data = data[data['future_return'] > 0.02]
    #print(filtered_data.shape[0])

def train_neural_network(training_data):
    """
    Create and train a neural network model based on the relevant data
    
    Parameters:
    - training_data: DataFrame object of the data we want to train the neural network on
    
    Returns:
    - a trained model
    """
    optimizer = Adam(learning_rate=0.0001)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(len(columns),)))
    #model.add(Dropout(0.5)) 
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5)) 
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(training_data[columns],
            training_data['direction'],
            epochs=50, 
            verbose=False,)
            #validation_split=0.2,
            #shuffle = False)

    return model

def calculate_position_percentage(data, investment_percentage, position_size):

    portfolio_investment_percentage = investment_percentage
    position_size_percentage = position_size
    current_position_percentage = 0
    max_investment_limit = 1.0

    for index, row in data.iterrows():
        signal = row['prediction']
        
        # Check if adding another position exceeds the portfolio limit
        if (signal == 1) & (portfolio_investment_percentage + position_size_percentage <= max_investment_limit):

            #if current_position_percentage<0:
            #    current_position_percentage = 0
            current_position_percentage += position_size_percentage
            portfolio_investment_percentage += position_size_percentage

        elif (portfolio_investment_percentage-position_size_percentage>= -max_investment_limit):
            if signal == -1:
                #if current_position_percentage>0:
                #    current_position_percentage = 0
                current_position_percentage -= position_size_percentage
                portfolio_investment_percentage = portfolio_investment_percentage - position_size_percentage
        
        # Store the updated position size percentage for each row
        data.at[index, 'current_position_percentage'] = current_position_percentage
        data.at[index, 'portfolio_investment_percentage'] = portfolio_investment_percentage
    

def plot(strategy_and_returns_columns, buy_signals, sell_signals):
    plt.figure(figsize=(16, 10))
    strategy_and_returns_columns.plot(ax=plt.gca())

    # buy trades with green triangles and sell trades with red triangles
    # assuming '1' is a buy signal and '-1' is a sell signal

    plt.scatter(buy_signals.index, buy_signals, color='green', marker='^', label='Buy Trades', alpha=0.7)
    plt.scatter(sell_signals.index, sell_signals, color='red', marker='v', label='Sell Trades', alpha=0.7)
    #close_signal_values = cumulative_returns.loc[valid_close_signals, 'strategy']
    #plt.scatter(close_signal_values.index, close_signal_values, color='blue', marker='x', label='Close Signals')

    plt.title('Trading Strategy Performance with Trade Signals')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

if __name__ == "__main__":   
    data = prepare_training_data('SOLUSDT-1h-2023-01.csv')
    add_indicators_to_training_data(data, columns, 'close')

    training_data = data.copy()
    mu, std = training_data.mean(), training_data.std()
    training_data_ = (training_data-mu)/ std

    model = train_neural_network(training_data)
    model.save('my_model.h5')
    #res = pd.DataFrame(model.history.history)

    model.evaluate(training_data_[columns], training_data['direction'])
    predictions = model.predict(training_data_[columns])
    predictions_series = pd.Series(predictions.flatten(), index=training_data.index)

    training_data['prediction'] = np.where(predictions_series > buy_threshold, 1, np.where(predictions_series < sell_threshold, -1, 0))
    #training_data['prediction'] = np.where(predictions > buy_threshold, 1, np.where(predictions < sell_threshold, -1, 0))

    training_data['position'] = 0  # Initialize position column
    last_position= 0

    for index, row in training_data.iterrows():
        current_prediction = row['prediction']
        
        # Directly get the scalar value for the current index's prediction
        pred_value = predictions_series.loc[index]
        if isinstance(pred_value, pd.Series):
            pred_value = pred_value.iloc[0]  # Ensuring a scalar value
        
        # Buy signal
        if current_prediction == 1:
            last_position = 1
        # Sell signal
        elif current_prediction == -1:
            last_position = -1
        # Evaluate closing conditions
        """
        else:
            # Close long position based on the close_long_threshold
            if last_position == 1 and pred_value < close_long_threshold:
                last_position = 0
            # Close short position based on the close_short_threshold
            elif last_position == -1 and pred_value > close_short_threshold:
                last_position = 0
        """
        
        # Correctly apply the last_position to the current row
        training_data.at[index, 'position'] = last_position

    training_data['strategy'] = training_data['prediction'] * training_data['returns']

    calculate_position_percentage(training_data, 0, 1.0)

    training_data['strategy_returns'] = training_data['returns'] * training_data['current_position_percentage']

    # Convert strategy log returns to simple returns
    training_data['strategy_simple_returns'] = np.exp(training_data['strategy_returns']) - 1

    # Calculate cumulative simple returns and portfolio value
    #training_data['cumulative_strategy_returns'] = (1 + training_data['strategy_simple_returns']).cumprod()
    #training_data['portfolio_value'] = 1 * training_data['cumulative_strategy_returns']
    # cumulative returns for the strategy and the underlying asset
    cumulative_returns = training_data[['returns', 'strategy_returns']].cumsum().apply(np.exp)

    buy_signals = training_data[training_data['prediction'] == 1].index
    sell_signals = training_data[training_data['prediction'] == -1].index
    close_signals = training_data[training_data['position'] == 0].index

    # Filter buy and sell signals to ensure they exist in cumulative_returns
    valid_buy_signals = [signal for signal in buy_signals if signal in cumulative_returns.index]
    valid_sell_signals = [signal for signal in sell_signals if signal in cumulative_returns.index]
    valid_close_signals = [signal for signal in close_signals if signal in cumulative_returns.index]

    # Convert filtered signals back to a pandas Index or Series (if needed)
    valid_buy_signals = pd.Index(valid_buy_signals)
    valid_sell_signals = pd.Index(valid_sell_signals)
    valid_close_signals = pd.Index(valid_close_signals)
    #print(data.info())
    buy_signal_values = cumulative_returns.loc[valid_buy_signals, 'strategy_returns']
    sell_signal_values = cumulative_returns.loc[valid_sell_signals, 'strategy_returns']

    # Debugging: Print lengths to check
    print(f"Number of valid buy signals: {len(valid_buy_signals)}")
    print(f"Number of valid sell signals: {len(valid_sell_signals)}")
    print(f"Number of valid close signals: {len(valid_close_signals)}")
    #print(training_data.loc[training_data['prediction']==-1, 'current_position_percentage'])
    # plot cumulative returns

    plot(cumulative_returns, buy_signal_values, sell_signal_values)
