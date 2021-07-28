import os
import pandas as pd
from glob import glob
import pickle
from fbprophet import Prophet
from matplotlib import pyplot as plt
import seaborn as sns

import pickle

import numpy as np

from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans

bull_market_prices = pd.read_csv('../data/bull-market.csv')
prices_2008 = pd.read_csv('../data/2008-market.csv')
prices_2018 = pd.read_csv('../data/2018-market.csv')

def get_prices(ticker, year='2010'):
    if year == '2008':
        return prices_2008[prices_2008['tic'] == ticker]
    elif year == '2018':
        return prices_2018[prices_2018['tic'] == ticker]
    else:
        return bull_market_prices[bull_market_prices['tic'] == ticker]


folders = os.listdir('./results')
print(folders)

for folder in folders:
    if os.path.isdir('./results/' + folder):
        seasonalities = glob('./results/' + folder + '/*-seasonalities.p')
        models = glob('./results/' + folder + '/*-model.p')
        for i, seasonality in enumerate(seasonalities):
            with open(seasonality, 'rb') as f:
                data = pickle.load(f)

            clf = TimeSeriesKMeans().from_pickle(models[i])
            tickers = list(data.keys())
            industries = []
            for ticker in tickers: 
                industries.append(get_prices(ticker, seasonality.split('-')[1]).iloc[0]['gind'])
            dataset = []
            for value in data:
                dataset.append(data[value]['seasonality_vector'])
            X = to_time_series_dataset(dataset)
            X = TimeSeriesScalerMeanVariance().fit_transform(X)
            y = clf.predict(X)
            clusters = {}
            for yi in range(4):
                clusters[yi] = []
                for xx in X[y == yi]:
                    i = np.where(X == xx)[0][0]
                    clusters[yi].append(industries[i])
            fig = plt.figure(figsize=(20, 20))
            split_file = seasonality.split('-')
            for yi in range(4):
                plt.subplot(3, 3, yi + 1)
                if yi == 1:
                    title = plt.title("Industry Code by Cluster for Sector {} ({})".format(split_file[0].split('/')[1], split_file[1]))
                    title.set_position([.5, 1.05])
                    plt.xlabel('Industrial Code')
                    plt.ylabel('Count')
                plt.xticks(rotation=45, ha="right")
                sns.countplot(clusters[yi], palette="RdBu")
            fig.savefig('{}-{}-industries-by-cluster.png'.format(split_file[0], split_file[1]), bbox_inches='tight', dpi=250)
    else:
        continue