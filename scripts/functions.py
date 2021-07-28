import pandas as pd
import pandas as pd
import numpy as np

from fbprophet import Prophet
from fbprophet.plot import plot_yearly

bull_market_prices = pd.read_csv('../data/bull-market.csv')
prices_2008 = pd.read_csv('../data/2008-market.csv')
prices_2018 = pd.read_csv('../data/2018-market.csv')

market_capitalizations = pd.read_csv('../data/market-capitalization.csv')
market_capitalizations = market_capitalizations[['datadate', 'tic', 'csho']]
market_capitalizations.loc[:, 'datadate'] = pd.to_datetime(market_capitalizations['datadate'], format='%m/%d/%Y')



def get_prices(ticker, year='2010'):
    if year == '2008':
        return prices_2008[prices_2008['tic'] == ticker]
    elif year == '2018':
        return prices_2018[prices_2018['tic'] == ticker]
    else:
        return bull_market_prices[bull_market_prices['tic'] == ticker]

def get_prices_by_sector(year='2010', gsector=45.0):
    if year == '2008':
        return prices_2008[prices_2008['gsector'] == gsector]
    elif year == '2018':
        return prices_2018[prices_2018['gsector'] == gsector]
    else:
        return bull_market_prices[bull_market_prices['gsector'] == gsector]


def format_prophet_data(df):
    new_df = df[['datadate', 'prccd']].copy()
    new_df = new_df.rename(columns={'datadate': 'ds', 'prccd': 'y'})
    new_df['ds'] = pd.to_datetime(new_df['ds'], format='%m/%d/%Y')
    return new_df


def compute_yearly_seasonality(df):
    m = Prophet(yearly_seasonality=10, weekly_seasonality=3, daily_seasonality=False, seasonality_mode='additive')
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    return forecast['yearly'][:df.shape[0]]


def get_market_cap(ticker, year=2010):
    ticker_shares = market_capitalizations[market_capitalizations['tic'] == ticker]
    shares_outstanding = ticker_shares[ticker_shares['datadate'].dt.year == year]['csho']
    ticker_price = get_prices(ticker, year=str(year)).iloc[0]['prccd']
    return (ticker_price * shares_outstanding).values[0]




def get_sample(market_caps):
    vals = list(market_caps.values())
    vals.sort()
    divider = len(vals) // 3
    small_caps = vals[:divider]
    mid_caps = vals[divider:divider * 2]
    large_caps = vals[divider * 2:]
    sample = []
    
    random_small_caps = list(np.random.choice(small_caps, 100))
    random_mid_caps = list(np.random.choice(mid_caps, 100))
    random_large_caps = list(np.random.choice(mid_caps, 100))
    sample_set = random_small_caps + random_mid_caps + random_large_caps 
    
    for cap in sample_set:
        for key in market_caps:
            if market_caps[key] == cap:
                if cap in random_small_caps:
                    sample.append((key, 'small'))
                elif cap in random_mid_caps:
                    sample.append((key, 'mid'))
                else:
                    sample.append((key, 'large'))
                break
    return sample

def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:  
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len:-window_len+1]