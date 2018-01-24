#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 22:56:52 2017

@author: BennyBluebird
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')

def get_data(airline, airport, categories=["Passengers"]):
    
    """Takes airline and airport code strings along with one or more data
    category strings as inputs and returns a pandas Series if only one category 
    is requested. Returns pandas DataFrame for calls with more than one category.
    """
    # Date indexes are read in as unicode literals, dataparse will properly
    # reconvert them to DatetimeIndex format
    
    data = pd.read_csv('../data/{}-{}.csv'.format(airline, airport), index_col='Date', 
                       parse_dates=True, date_parser=dateparse)
    
    # Returns DataFrame if more than one category is requested
    if len(categories) > 1:
        columns = ['{}_Domestic'.format(category) for category in categories]
        return data[columns].astype(np.float64)
    
    # Returns Series if only one category is requested
    else:
        return data['{}_Domestic'.format(categories[0])].astype(np.float64)


def test_stationarity(time_series):
    
    """Takes a single pandas Series and produces evidence that can be used
    to analyze stationarity or lack thereof in the time series. Will not work
    with pandas DataFrame, numpy array, or any other data format.
    """

    # Check for upward or downward sloping trends in the moving average.
    # Trends indicate non-stationarity which should be taken into account
    # when building ARIMA model.
    
    moving_average = time_series.rolling(window=12).mean()
    moving_std = time_series.rolling(window=12).std()
    name = time_series.name.split('_')[0]
    
    plt.plot(time_series, color='blue', label='Monthly {}'.format(name))
    plt.plot(moving_average, color='red', label='Moving Average')
    plt.plot(moving_std, color='black', label='Moving Std.')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # The greater the p-value in the test output, the stronger the
    # non-stationarity of the time series. Series with p-value less
    # than 0.05 can generally be considered at least weakly stationary
    
    print('Results of Dickey-Fuller Test:')
    test = adfuller(time_series, autolag='AIC')
    test_output = pd.Series(test[0:4], index=['Test Statistic', 'p-value',
                         '#Lags Used', 'Number of Observations Used'])
    for key, value in test[4].items():
        test_output['Critical Value {}'.format(key)] = value
    print(test_output)
    
def _remove_seasonality(dataset, lag=12):
    
    # To be called inside predict_final year only. 
    
    difference = list()
    for i in range(lag, len(dataset)):
        value = dataset[i] - dataset[i - lag]
        difference.append(value)
    return np.array(difference)

def _add_seasonality(history, pred, lag=1):
    return pred + history[-lag]

def predict_final_year(time_series, order=(12,1,2), search=False):
    
    data = time_series.values
    train, test = data[:-12], data[-12:]
    differenced = _remove_seasonality(train, lag=12)
    model = ARIMA(differenced, order=order)
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(12)[0]
    history = [x for x in train]
    for pred in forecast:
        reverted = _add_seasonality(history, pred, lag=12)
        history.append(reverted)
    preds = np.array(history[-12:])
    
    # Only to be used when called from grid_search function. 
    # Should not be activated manually in any other context.
    if search:
        return mean_squared_error(test, preds)
    
    print('RMSE: ' + str(round(np.sqrt(mean_squared_error(test, preds)),3)))
    print('R_SQ: '+ str(round(r2_score(test, preds),3)))
    
    return test, preds
    
def grid_search(dataset, p_values=range(13), d_values=range(3), q_values=range(3)):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    print 'Testing ARIMA: {}'.format(order)
                    mse = predict_final_year(dataset, order, search=True)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print 'MSE: {:.3f}\n'.format(mse)
                except:
                    continue
    print 'Best ARIMA: {}'.format(best_cfg)
    print 'Best RMSE: {}'.format(np.sqrt(best_score)) 