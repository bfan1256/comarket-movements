import warnings
warnings.filterwarnings('ignore')


import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import pickle
from functions import *


from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans

import matplotlib.pyplot as plt
import seaborn as sns


bull_market_prices = pd.read_csv('../data/bull-market.csv')
gsectors = bull_market_prices['gsector'].unique()

for sector in gsectors:
    if not os.path.exists('./results/{}'.format(str(sector))):
        os.makedirs('./results/{}'.format(str(sector)))
    for year in ['2008', '2010', '2018']:
        prices = get_prices_by_sector(year=year, gsector=sector)[['datadate', 'tic', 'prccd']]
        all_tickers = prices['tic'].unique()
        market_caps = {}
        for ticker in tqdm(all_tickers):
            try:
                market_caps[ticker] = get_market_cap(ticker)
            except:
                continue
        with open('./results/{}/market_caps.p'.format(str(sector)), 'wb') as f:
            pickle.dump(market_caps, f)
        sample = get_sample(market_caps)
        data = {}
        for ticker, size in tqdm(sample):
            price = prices[prices['tic'] == ticker]
            prophet = format_prophet_data(price)
            try:
                seasonality = compute_yearly_seasonality(prophet)
            except:
                continue
            data[ticker] = {'symbol': ticker, 'size': size, 'seasonality_vector': seasonality.values} 
        with open('./results/{}/{}-{}-seasonalities.p'.format(str(sector), str(sector), year), 'wb') as f:
            pickle.dump(data, f)

        dataset = []
        for value in data:
            dataset.append(data[value]['seasonality_vector'])    
        X = to_time_series_dataset(dataset)
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
        clf = TimeSeriesKMeans(n_clusters=4, metric='dtw', max_iter=10, n_jobs=3)
        clf.fit(X)
        y = clf.predict(X)

        sz = X.shape[1]
        fig = plt.figure(figsize=(20, 20))
        for yi in range(4):
            plt.subplot(3, 3, yi + 1)
            for xx in X[y == yi]:
                plt.plot(xx.ravel(), "k-", alpha=.1)
            plt.plot(smooth(clf.cluster_centers_[yi].ravel()), "b-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.80, 0.95,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                if year == '2010':
                    plt.title("Sector {} During Bullish Market $k$-means (DTW)".format(str(sector)))
                    
                else:
                    plt.title("Sector {} During {} Bearish Market $k$-means (DTW)".format(str(sector), year))
                plt.xlabel('Time')
                plt.ylabel('Seasonality Indication')

        fig.savefig('./results/{}/{}-{}-market-clustering.png'.format(str(sector), str(sector), year), bbox_inches='tight', dpi=250)
        clf.to_pickle('./results/{}/{}-{}-model.p'.format(str(sector), str(sector), year))
        tickers = list(data.keys())
        fig = plt.figure(figsize=(20, 20))
        for yi in range(4):
            plt.subplot(3, 3, yi + 1)
            is_first_small = False
            is_first_mid = False
            is_first_large = False
            for xx in X[y == yi]:
                i = np.where(X == xx)[0][0]
                
                if data[tickers[i]]['size'] == 'small':
                    if not is_first_small and yi == 1:
                        plt.plot(xx.ravel(), "k-", alpha=0.5, label='Small Cap')
                        is_first_small = True
                    else:
                        plt.plot(xx.ravel(), "k-", alpha=.1)
                elif data[tickers[i]]['size'] == 'mid':
                    if not is_first_mid and yi == 1:
                        plt.plot(xx.ravel(), 'g-', alpha=0.5, label='Mid Cap')
                        is_first_mid = True
                    else:
                        plt.plot(xx.ravel(), 'g-', alpha=.1)
                else:
                    if not is_first_large and yi == 1:
                        plt.plot(xx.ravel(), 'r-', alpha=0.5, label='Large Cap')
                        is_first_large = True
                    else:
                        plt.plot(xx.ravel(), "r-", alpha=.1)
            plt.plot(smooth(clf.cluster_centers_[yi].ravel()), "b-")
            plt.xlim(0, sz)
            plt.ylim(-4, 4)
            plt.text(0.05, 0.95,'Cluster %d' % (yi + 1),
                    transform=plt.gca().transAxes)
            if yi == 1:
                if year == '2010':
                    title = plt.title("{} Sector Colored by Market Capitalization (2010-2012)".format(str(sector)))
                else:
                    title = plt.title("{} Sector Colored by Market Capitalization ({})".format(str(sector), year))
                title.set_position([.5, 1.05])
                plt.xlabel('Time')
                plt.ylabel('Seasonality Indication')
                plt.legend()
            fig.savefig('./results/{}/{}-sector-{}-market-by-market-cap.png'.format(str(sector), str(sector), str(year)), bbox_inches='tight', dpi=250)