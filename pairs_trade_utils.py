import datetime
import warnings
import pickle

import numpy as np
from scipy import stats
import pandas as pd
import yfinance as yfinance

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import coint, adfuller

from pykalman import KalmanFilter

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas_datareader as pdr
import scipy.stats as s

warnings.filterwarnings('ignore')

in_sample_start = '2018-01-01'
out_sample_start = '2021-01-01'
out_sample_end = '2023-03-01'

def get_closing_prices(tickers, date_start, date_end):
    """
    Retrieve closing prices for specified tickers within a given date range.

    :param tickers: List of stock tickers
    :type tickers: list
    :param date_start: Start date for fetching historical prices
    :type date_start: str
    :param date_end: End date for fetching historical prices
    :type date_end: str
    :return: DataFrame containing adjusted closing prices for specified tickers
    :rtype: pd.DataFrame
    """
    prices = yf.download(tickers, start=date_start, end=date_end)['Adj Close']
    return prices.dropna(axis=1)


def get_volume(tickers, date_start, date_end):
    """
    Retrieve trading volumes for specified tickers within a given date range.

    :param tickers: List of stock tickers
    :type tickers: list
    :param date_start: Start date for fetching historical volumes
    :type date_start: str
    :param date_end: End date for fetching historical volumes
    :type date_end: str
    :return: DataFrame containing trading volumes for specified tickers
    :rtype: pd.DataFrame
    """
    volumes = yf.download(tickers, start=date_start, end=date_end)['Volume']
    return volumes.dropna(axis=1)


def split(df, in_sample_start, out_sample_start):
    """
    Split a DataFrame into in-sample and out-of-sample periods based on specified dates.

    :param df: DataFrame to be split
    :type df: pd.DataFrame
    :param in_sample_start: Start date of the in-sample period
    :type in_sample_start: str
    :param out_sample_start: Start date of the out-of-sample period
    :type out_sample_start: str
    :return: Two DataFrames representing the in-sample and out-of-sample periods
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    in_sample = df[(df.index >= pd.to_datetime(in_sample_start)) & (df.index <= pd.to_datetime(out_sample_start))]
    out_of_sample = df[df.index >= pd.to_datetime(out_sample_start)]
    return in_sample, out_of_sample


def rank_cointegration(df, sector_map):
    """
    Rank cointegration between pairs of time series.

    :param df: DataFrame containing time series data
    :type df: pd.DataFrame
    :param sector_map: Dictionary mapping tickers to their respective sectors
    :type sector_map: dict
    :return: Two DataFrames - one containing cointegration p-values, and another with pair information
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    n, cols = df.shape[1], df.keys()
    vals, pairs, ls, rs = [], [], [], []
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            a, b = df[cols[i]], df[cols[j]]
            val = coint(a, b)[1]
            pair = cols[i] + ' ' + cols[j]
            l, r = sector_map[cols[i]], sector_map[cols[j]]
            vals.append(val)
            pairs.append(pair)
            ls.append(l)
            rs.append(r)
            matrix[i,j] = val    
    matrix = matrix.T + matrix
    np.fill_diagonal(matrix, 1)
    vals_df = pd.DataFrame(matrix, columns=df.columns, index=df.columns)
    pairs_df = pd.DataFrame({'pval': vals, 'left': ls, 'right': rs}, index=pairs).sort_values(by='pval')
    return pairs_df, vals_df


def group_indices(df, index):
    df.index = index
    df.columns = index
    return df


def sharpe(return_series, n, rf):
    mean = (return_series.mean() * n) - rf
    sigma = return_series[30:].std() * np.sqrt(n) 
    return (mean/sigma).round(3)

def ir(return_series, bm_return, n):
    return_series = return_series - bm_return
    mean = (return_series.mean() * n)
    sigma = return_series[30:].std() * np.sqrt(n) 
    return (mean/sigma).round(3)


def KFHedgeRatio(x, y):
    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,initial_state_mean=[0, 0],
                      initial_state_covariance=np.ones((2, 2)),
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      observation_covariance=2,
                      transition_covariance=trans_cov)

    state_means, _ = kf.filter(y.values)
    return pd.DataFrame(state_means)[0]

def z_score(spread, s=5, l=30):
 
    l_spread = spread.rolling(window=l)
    s_spread = spread.rolling(window=s).mean()
    z_spread = s_spread.sub(l_spread.mean()).div(l_spread.std())
    return z_spread

def calc_spread(x, y):
    hr = KFHedgeRatio(x,y).values
    spread = y - hr * x
    return spread, hr

def create_df(x, y, s=5, l=30):

    spread, hr = calc_spread(x,y)
    z_spread = z_score(spread, s, l)
    
    df = pd.DataFrame({
                        f'y': y, 
                        f'x': x,
                        'hr': hr,
                        's' : spread,
                        'z': z_spread       
                    })
    df['trade_type'] = 0
    df['portfolio_val'] = 0

    return df
    
def simulate_trades(df, upper=0.5, lower=-0.5):
    holding, top_leg, bottom_leg, entries, exits = False, False, False, 0, 0
    for i in range(1, len(df)):
        if not holding:
            if df.z[i] >= upper:
                entry = df.y[i] - df.x[i] * df.hr[i] 
                holding, top_leg = True, True
                entries += 1
                df.iat[i, -2] = 'upper'
            elif df.z[i] <= lower: 
                entry = -(df.y[i] - df.x[i] * df.hr[i])
                holding, bottom_leg = True, True
                entries += 1
                df.iat[i, -2] = 'lower'
            else:
                entry = 0
            df.iat[i, -1] = entry
        else: 
            if top_leg == True and df.z[i] <= 0:
                exit =  -(df.y[i] - df.x[i] * df.hr[i])
                top_leg, holding = False, False
                df.iat[i, -2] = 'exit'
                exits += 1
            elif bottom_leg == True and df.z[i] >= 0:
                exit = (df.y[i] - df.x[i] * df.hr[i]) 
                bottom_leg, holding = False, False
                exits += 1
                df.iat[i, -2] = 'exit'
            else:
                exit = 0
            df.iat[i, -1] = exit 

    portfolio_val = (df.portfolio_val.cumsum()) + df.hr[0] * df.x[0]
    df.portfolio_val = portfolio_val
    rets = np.log(portfolio_val) - np.log(portfolio_val.shift(1))
    df['rets'] = rets
    df['cum_rets'] = rets.cumsum()

    print(f'Number of entry trades: {entries}')
    print(f'Number of exit trades: {exits}')
    print(f'Average time in days till entry or exit trade: {round(df.shape[0]/(entries+exits),3)}')

    return df, portfolio_val, rets


def test_sharpe_all_pairs(pairs, data, bm_rets, upper=0.5, lower=-0.5):
    """
    Test Sharpe ratios and Information Ratios for all pairs in a given dataset.

    :param pairs: DataFrame with cointegration pairs and sector information
    :type pairs: pd.DataFrame
    :param data: DataFrame containing time series data for all tickers
    :type data: pd.DataFrame
    :param bm_rets: Benchmark returns for comparison
    :type bm_rets: pd.Series
    :param upper: Upper threshold for trading signals, defaults to 0.5
    :type upper: float, optional
    :param lower: Lower threshold for trading signals, defaults to -0.5
    :type lower: float, optional
    :return: Two DataFrames - one containing Sharpe ratios and Information Ratios, and another with cumulative returns
    :rtype: tuple (pd.DataFrame, pd.DataFrame)
    """
    sharpe_ratios, irs = [], []
    rets_df = pd.DataFrame(index=data.index)
    
    for pair in pairs.index.tolist():
        s1, s2 = pair.split()   
        df = create_df(data[s1], data[s2])
        _,_,rets = simulate_trades(df, upper=upper, lower=lower)
        sharpe_ratio = sharpe(rets, 252, 0.02)
        sharpe_ratios.append(sharpe_ratio)
        info_ratio = ir(rets, bm_rets, 252)
        irs.append(info_ratio)
        rets_df[pair] = rets
    
    sharpes = pd.DataFrame({'sharpe': sharpe_ratios, 'ir': irs}, index=pairs.index).sort_values(by='sharpe', ascending=False)
    rets_df = rets_df.fillna(0).cumsum()
    rets_df = rets_df[sharpes.index]
    
    return sharpes, rets_df


def raw_performance(stock_pairs, cutoff, stocks, ndx, in_sample=False):
    """
    Calculate raw performance metrics for selected stock pairs.

    :param stock_pairs: DataFrame containing cointegration pairs and sector information
    :type stock_pairs: pd.DataFrame
    :param cutoff: Cutoff threshold for selecting stock pairs based on cointegration p-values
    :type cutoff: float
    :param stocks: DataFrame containing time series data for all tickers
    :type stocks: pd.DataFrame
    :param ndx: Benchmark returns for comparison
    :type ndx: pd.Series
    :param in_sample: Flag indicating whether the analysis is performed in-sample, defaults to False
    :type in_sample: bool, optional
    """
    pairs = stock_pairs
    if cutoff <= 0.5:
        pairs = pairs[pairs.pval <= cutoff]
    else:
        pairs = pairs[pairs.pval >= cutoff]
    same_sector_pairs = pairs[pairs.left == pairs.right]
    sharpes, rets_df = test_sharpe_all_pairs(pairs, stocks, ndx)
    _sharpes, _rets_df = test_sharpe_all_pairs(same_sector_pairs, stocks, ndx)
    sharpes = sharpes.merge(stock_pairs, how='left', left_on=sharpes.index, right_on=stock_pairs.index)
    _sharpes = _sharpes.merge(stock_pairs, how='left', left_on=_sharpes.index, right_on=stock_pairs.index)
    sharpes.set_index('key_0', inplace=True)
    _sharpes.set_index('key_0', inplace=True)

    suffix, prefix = '', ''
    if in_sample:
        suffix = '_is'
    if cutoff >= 0.5:
        prefix = 'bad'