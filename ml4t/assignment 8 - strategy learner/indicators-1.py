'''
 Your code that implements your indicators as functions that operate on dataframes.
 The "main" code in indicators.py should generate the charts that illustrate your
 indicators in the report.

Student Name: Donald Ward (replace with your name)
GT User ID: dward45 (replace with your User ID)
GT ID: 903271210 (replace with your GT ID)
'''
import pandas as pd
import datetime as dt
import marketsimcode as marketsim
from matplotlib import pyplot as plt


def author():
    return 'dward45'  # replace tb34 with your Georgia Tech username.


def bb(df_prices, rolling):
    # adapted from: https://quant.stackexchange.com/questions/11264/calculating-bollinger-band-correctly
    num_std = 2
    rolling_mean = df_prices.rolling(rolling).mean()
    rolling_std = df_prices.rolling(rolling).std()
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)
    return rolling_mean, upper_band, lower_band


#-------------------------------------------------
# SMA NEVER USED IN MANUAL STRATEGY, OR IN QLEARNER
#-------------------------------------------------
# Sell when ratio above a value, Buy when below a value
def sma(df_prices, rolling):
    # adapted from: https://stackoverflow.com/questions/39138299/dataframe-sma-calculation
    df_sma = df_prices.rolling(rolling).mean()
    return df_sma


def mom(df_prices, rolling):
    df_prices_older = df_prices.shift(rolling)
    return (df_prices / df_prices_older) - 1


def norm(df):
    df_tmp = df.copy()
    df_tmp.fillna(method='ffill', inplace=True)
    df_tmp.fillna(method='bfill', inplace=True)
    return df_tmp / df_tmp.ix[0, :]


def macd(prices_df, newer=12, older=26):
    '''
    Adapted from https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
    '''
    df = prices_df.copy()
    prices = pd.Series(prices_df.ix[:, 0], index=prices_df.index)
    fast = pd.Series(pd.ewma(prices, span=newer, min_periods=older - 1))
    slow = pd.Series(pd.ewma(prices, span=older, min_periods=older - 1))
    MACD = pd.Series(fast - slow, name='MACD')
    MACDsign = pd.Series(pd.ewma(MACD, span=9, min_periods=8), name='MACDsign')
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff')

    df = df.join(MACD)
    df = df.join(MACDsign)
    #df = df.join(MACDdiff)
    df = df.drop(df.columns[0], axis=1)
    return df


def generate_report_graphs():
    symbol = "JPM"
    rolling = 20
    sd = dt.datetime(2008, 1, 1)
    adj_sd = dt.datetime(2007, 11, 15)
    ed = dt.datetime(2009, 12, 31)
    # rbd = sd - dt.timedelta(rolling)
    sv = 100000

    print "Symbol: ", symbol
    print "Start date: ", sd
    print "End date: ", ed
    print "Starting cash:", sv
    print "Rolling days: ", rolling

    # Acquire prices for date range and normalize
    prices, prices_SPY = marketsim.get_data_by_column([symbol], adj_sd, ed, marketsim.DATACOL_ADJUSTED_CLOSE)

    # Get SMA Indicator
    sma_jpm = sma(prices, rolling)
    prices_sma_ratio = prices / sma_jpm

    # Get BB Indicator
    bb_mean, bb_upper, bb_lower = bb(prices, rolling)
    bb_ratio = (prices - bb_lower) / (bb_upper - bb_lower)

    # Get MOM Indicator
    mom_days = 14
    mom_jpm = mom(prices, mom_days)

    # Get MACD indicator
    macd_jpm = macd(prices)



    #
    # ===== Plotting =======
    #

    # Plot BB
    bb_plot = pd.concat([norm(prices), norm(bb_lower), norm(bb_upper)], axis=1)
    bb_plot = bb_plot[sd:ed]
    bbrat_plot = pd.concat([bb_ratio], axis=1)
    bbrat_plot = bbrat_plot[sd:ed]

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    bb_plot.columns = [symbol, 'Lower Band', 'Upper Band']
    bb_plot.plot(grid=True, title=symbol + ' Price, Bollinger Bands and BB %', use_index=True, ax=ax1)

    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    bbrat_plot.columns = ['BB %']
    bbrat_plot.plot(grid=True, title='Bollinger Percent', use_index=True, ax=ax2, sharex=True)

    ax2.set_xlabel('Date')
    ax1.set_ylabel('Normed Price')
    ax2.set_ylabel('Normed BB%')

    plt.tight_layout()
    plt.savefig("indicators-bollinger-bands.png", format='png', bbox_inches="tight")

    # Plot MOM
    mom_plot = pd.concat([prices, norm(mom_jpm)], axis=1)
    mom_plot = mom_plot[sd:ed]
    mom_plot.columns = [symbol, 'Momentum']
    mom_plot.plot(grid=True, title=symbol + " Price and Momentum", use_index=True)
    plt.ylabel("Price and Momentum")
    plt.xlabel("Date")

    plt.tight_layout()
    plt.savefig("indicators-momentum.png", format='png', bbox_inches="tight")

    # Plot SMA

    sma_plot = pd.concat([norm(prices), norm(sma_jpm)], axis=1)
    sma_plot = sma_plot[sd:ed]
    prices_sma_ratio_plot = pd.concat([prices_sma_ratio], axis=1)
    prices_sma_ratio_plot = prices_sma_ratio_plot[sd:ed]

    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    sma_plot.columns = [symbol, 'SMA']
    sma_plot.plot(grid=True, title=symbol + ' Price, SMA', use_index=True, ax=ax1)

    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    prices_sma_ratio_plot.columns = ['Price/SMA Ratio']
    prices_sma_ratio_plot.plot(grid=True, use_index=True, ax=ax2, sharex=True)

    ax2.set_xlabel('Date')
    ax1.set_ylabel('Normed Values')
    ax2.set_ylabel('Normed Ratio')

    plt.tight_layout()
    plt.savefig("indicators-sma.png", format='png', bbox_isnches="tight")



    # Plot macd
    prices_plot = pd.concat([prices], axis=1)
    macd_jpm_plot = pd.concat([macd_jpm], axis=1)
    macd_jpm_plot = macd_jpm_plot[sd:ed]

    macd_jpm_plot = pd.concat([macd_jpm], axis=1)
    macd_jpm_plot = macd_jpm_plot[sd:ed]
    macd_jpm_plot.columns = ['MACD', 'Signal']
    macd_jpm_plot.plot(grid=True, title="MACD and MACD Sign", use_index=True)
    # plt.ylabel("Normed Values")
    # plt.xlabel("Date")


    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    prices_plot.columns = [symbol]
    prices_plot.plot(grid=True, title=symbol + ' Price', use_index=True, ax=ax1)

    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=2)
    macd_jpm_plot.columns = ['MACD', 'MACD Signal']
    macd_jpm_plot.plot(grid=True, title='MACD and Signal', use_index=True, ax=ax2, sharex=True)

    ax1.set_ylabel('Price')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig("indicators-macd.png", format='png')


if __name__ == '__main__':
    generate_report_graphs()
