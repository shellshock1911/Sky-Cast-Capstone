# Compatibility between Python 2.7 and 3.x
from __future__ import unicode_literals, print_function, division
from io import open

# Wrangling
import glob # Files
import os # Files
import numpy as np # Arrays
import pandas as pd # DataFrames

# Preprocessing
from sklearn.preprocessing import StandardScaler # Centers mean at zero, applies unit variance
# Visualization
import matplotlib.pyplot as plt
from matplotlib import style
from statsmodels.tsa.stattools import adfuller
style.use('fivethirtyeight')

DATES = pd.date_range(start='2016-04-01', end='2017-03-01', freq='MS').to_pydatetime()

# Mapping of IATA airline codes to airline name
# Taken from http://www.iata.org/publications/Pages/code-search.aspx
AIRLINE_CODES = {
    'AS': 'Alaska_Airlines',
    'G4': 'Allegiant_Air',
    'AA': 'American_Airlines',
    'DL': 'Delta_Airlines',
    'F9': 'Frontier_Airlines',
    'MQ': 'Envoy_Air',
    'EV': 'ExpressJet_Airlines',
    'HA': 'Hawaiian_Airlines',
    'B6': 'JetBlue_Airways',
    'OO': 'SkyWest_Airlines',
    'WN': 'Southwest_Airlines',
    'NK': 'Spirit_Airlines',
    'UA': 'United_Airlines',
}

def load_data(airlines, category):
    """
    Takes a list of airlines and an industry metric and returns
    a N X 174 dataframe, where N rows represents the number of unique
    airline-airport combinations for which data exists and 174 columns
    represents the maximum number of months in the datasets
    
    @airlines: List of IATA airline codes (AA, DL, UA, etc.)
    @category: Industry metric to slice (Passengers, Flights, ASM, RPM)
    """
    
    rows = list() # Hold rows for data on each airline-airport combination
    index = list() # A combination (e.g. AA-DFW, DL-ATL, UA-ORD) serves as the row label
    
    # Monthly intervals between October 2002 to March 2017
    dates = [pd.datetime.strftime(date, '%Y-%m') for date in 
             pd.date_range(start='2002-10-01', end='2017-03-31', freq='M')]
    
    # Go into each airline directory
    for airline in airlines:
        # Make a list of all airline-airport csv files
        datasets = glob.glob('../datasets/{}/*.csv'.format(AIRLINE_CODES[airline]))
        # Then load each dataset and slice the desired industry metric
        for dataset in datasets:
            data = pd.read_csv(dataset)
            series = data[category + '_Domestic']
            # Ensure that data for all 174 months can be fetched
            if len(data) == 174:
                # Make dictionary with dates as keys and metric quantities as values
                # Values for each column label (dates) now represent a single row 
                row = {date: count for date, count in zip(dates, series)}
                # Add to row list
                rows.append(row)
                # Label row with airline-airport combination
                index.append(os.path.splitext(os.path.split(dataset)[1])[0])
    
    # Convert rows, dates, and index into DataFrame            
    df = pd.DataFrame(data=rows, columns=dates, index=index)
    df.index.name = category
    return df

def remove_seasonality(time_series, lag):
    difference = list()
    for i in range(lag, len(time_series)):
        value = time_series[i] - time_series[i - lag]
        difference.append(value)
    time_series = pd.Series(difference)
    return time_series

def add_seasonality(history, pred, lag=1):
    return pred + history[-lag]

DATES = pd.date_range(start='2002-10-01', end='2017-03-01', freq='MS').to_pydatetime()

def test_stationarity(df, airline, airport, lag, deseason=False):
    
    """Takes a single pandas Series and produces evidence that can be used
    to analyze stationarity or lack thereof in the time series
    """
    
    time_series = df.loc[airline + '-' + airport].reset_index(drop=True)
    
    if deseason:
        time_series = remove_seasonality(time_series, lag)
        
    # Check for upward or downward sloping trends in the moving average.
    # Trends indicate non-stationarity which should be taken into account
    # when building ARIMA model.
    
    moving_average = time_series.rolling(window=lag).mean()
    moving_std = time_series.rolling(window=lag).std()
    title = "Rolling Mean & Standard Deviation"
   
    plt.figure(figsize=(20, 8))
    plt.plot(DATES, time_series, color='blue', label='Monthly {0}-{1} {2}'.format(airline, airport, df.index.name))
    plt.plot(DATES, moving_average, color='red', label='{} Month Moving Average'.format(lag))
    plt.plot(DATES, moving_std, color='black', label='{} Month Moving Std.'.format(lag))
    plt.xlabel("Year")
    plt.ylabel("Monthly {}".format(df.index.name))
    plt.legend(loc='best')
    plt.title(title, fontsize=20)
    plt.setp(plt.xticks()[1], fontsize=14)
    plt.savefig('trend_visual-{0}-{1}'.format(airline, airport))
    plt.show()
    
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

def MAPE(y_test, preds):
    """Mean Absolute Percentage Error:
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    mape_score = 100 * (np.sum(np.abs(y_test - preds) / y_test) / len(y_test))
    return round(mape_score, 3)

def fetch_sample(df, batch_size, output_seq_len, excluded=None, random_state=None):
    """
    Takes a list of airlines and an industry metric and returns
    a N X 174 dataframe, where N rows represents the number of unique
    airline-airport combinations for which data exists and 174 columns
    represents the maximum number of months in the datasets
    
    @df: DataFrame to sample random time series from (Passengers, Flights, ASM, RPM)
    @batch_size: How many time series to sample at a time
    @output_seq_len: How many months to reserve for prediction (training target)
    @random_state: Controls reproducible output
    """
    
    # During training, exclude samples that will later be used for testing (to prevent overfitting)
    if excluded:
        excluded = df.index.isin(excluded)
        df = df[~excluded]
    
    # Train on all months prior to the months reserved for prediction
    input_seq_len = df.shape[1] - output_seq_len
    
    # Sample n random rows from the given dataframe
    sample = df.sample(n=batch_size, replace=False, axis=0, random_state=random_state)
    # Log transform values in batch
    log_sample = np.log(sample)
    
    # Standardize numerical values in batch to zero mean and unit variance
    scaler = StandardScaler()
    scaled_sample = scaler.fit_transform(log_sample.transpose())
    
    # Separate past values (training sequence input) from future values (target sequence output)
    sample_X, sample_y = scaled_sample[:input_seq_len], scaled_sample[-output_seq_len:]
    
    # Reshape dimensions to sequence_length X batch_size X 1 (univariate time series)
    sample_X = sample_X.reshape((input_seq_len, batch_size, 1))
    sample_y = sample_y.reshape((output_seq_len, batch_size, 1))
    
    return scaler, sample_X, sample_y

def load_test_series(df, airline, airport, output_seq_len):
    """
    Loads a specific airline-airport time series for testing, applies transformations,
    and divides it into test features and targets
    
    @df: DataFrame to draw specific time series from (Passengers, Flights, ASM, RPM)
    @airline: IATA airline code (AA, DL, UA, etc.)
    @airport: IATA airport code (ATL, DFW, ORD, etc.)
    @output_seq_len: How many months to reserve for prediction (test target)
    """
    # Train on all months prior to the months reserved for prediction
    
    input_seq_len = df.shape[1] - output_seq_len
    
    # Load specific time series given by airline-airport parameters
    series = df.loc[airline + '-' + airport]
    # Log transform series
    log_series = np.log(series)
    
    # Standardize series to zero mean and unit variance
    scaler = StandardScaler()
    scaled_series = scaler.fit_transform(log_series.values.reshape(-1, 1))
    scaled_series = scaled_series.reshape(-1)
    
    # Divide series into training 
    X_test = scaled_series[:input_seq_len]
    y_test = scaled_series[-output_seq_len:]
    
    # Return scaler to perform inverse transformation on data
    return scaler, X_test, y_test
