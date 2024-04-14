import pandas as pd
import numpy as np
from pylab import mpl, plt
from backtest import prepare_training_data, add_indicators_to_training_data, calculate_position_percentage
from keras.models import load_model

"""
This program applies the trading algorithm (neural network model)
created in the 'backtest.py' file onto a different dataset.

Here we can validate the model and test its performance with new data.

"""

columns = []
buy_threshold = 0.85
sell_threshold = 0.15

test_data = prepare_training_data('SOLUSDT-1h-2024-02.csv')
add_indicators_to_training_data(test_data, columns, 'close')

test_mu, test_std = test_data.mean(), test_data.std()
test_data_ = (test_data-test_mu)/ test_std

model = load_model('my_model.h5')

model.evaluate(test_data_[columns], test_data['direction'])
test_predict = model.predict(test_data_[columns])
test_predictions_series = pd.Series(test_predict.flatten(), index=test_data.index)
test_data['prediction'] = np.where(test_predict > buy_threshold, 1, np.where(test_predict < sell_threshold, -1, 0))
test_data['strategy'] = test_data['prediction'] * test_data['returns']

calculate_position_percentage(test_data, 0, 1.0)

test_data['strategy_returns'] = test_data['returns'] * test_data['current_position_percentage']
test_cumulative_returns = test_data[['returns', 'strategy_returns']].cumsum().apply(np.exp)

buy_signals = test_data[test_data['prediction'] == 1].index
sell_signals = test_data[test_data['prediction'] == -1].index
valid_buy_signals = [signal for signal in buy_signals if signal in test_cumulative_returns.index]
valid_sell_signals = [signal for signal in sell_signals if signal in test_cumulative_returns.index]
valid_buy_signals = pd.Index(valid_buy_signals)
valid_sell_signals = pd.Index(valid_sell_signals)

print(f"Number of valid buy signals: {len(valid_buy_signals)}")
print(f"Number of valid sell signals: {len(valid_sell_signals)}")

print(test_data.loc[test_data['prediction']==-1, 'current_position_percentage'])

plt.figure(figsize=(16, 10))
test_cumulative_returns.plot(ax=plt.gca())

buy_signal_values = test_cumulative_returns.loc[valid_buy_signals, 'strategy_returns']
plt.scatter(buy_signal_values.index, buy_signal_values, color='green', marker='^', label='Buy Trades', alpha=0.7)
sell_signal_values = test_cumulative_returns.loc[valid_sell_signals, 'strategy_returns']
plt.scatter(sell_signal_values.index, sell_signal_values, color='red', marker='v', label='Sell Trades', alpha=0.7)

plt.title('Trading Strategy Performance with Trade Signals')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
