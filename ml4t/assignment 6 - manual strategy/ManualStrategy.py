'''Code implementing a Manual object (details below). It should
implement testPolicy() which returns a trades data frame (see below). The main part
of this code should call marketsimcode as necessary to generate the plots used in
the report.
'''
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import marketsimcode as marketsim
from indicators import bb, mom, sma, macd
from matplotlib import pyplot as plt


def author():
    return 'dward45'  # replace tb34 with your Georgia Tech username.


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    """
    1) Trade with JPM, can use SPY to inform
    2) In sample dates: 1/1/2008 to 12/31/2009  Out sample: 1/1/2010 to 12/31/2011
    3) Starting cash 100,000
    4) 1000 long, 1000 short, 0 shares
    5) No limit on leverage
    6) Benchmark against 1000 shares of JPM and holding
    7) Transaction costs:
        ManualStrategy: Commission: $9.95, Impact: 0.005.
        TheoreticallyOptimalStrategy: Commission: $0.00, Impact: 0.00.
    """
    symbol = "JPM"
    rolling = 20
    adj_sd = sd - dt.timedelta(40) # to get rolling data

    # Acquire prices for date range and normalize
    prices, prices_SPY = marketsim.get_data_by_column([symbol], adj_sd, ed, marketsim.DATACOL_ADJUSTED_CLOSE)

    # Get SMA Indicator
    sma_df = sma(prices, rolling)
    prices_sma_ratio = prices / sma_df
    prices_sma_ratio = prices_sma_ratio[sd:ed]

    # Get BB Indicator
    bb_mean, bb_upper, bb_lower = bb(prices, rolling)
    # bb_ratio = pd.DataFrame(0, index=prices.index, columns=['BB Ratio'])
    bb_ratio_df = (prices - bb_lower) / (bb_upper - bb_lower)
    bb_ratio_df = bb_ratio_df[sd:ed]

    # Get MOM Indicator
    mom_days = 14
    mom_df = mom(prices, mom_days)
    mom_df = mom_df[sd:ed]

    # Get MACD Indicator
    macd_info_df = macd(prices)

    # Strip off earlier dates used for rolling data
    prices = prices[sd:ed]
    df = pd.DataFrame(0, index=prices.index, columns=['Shares'])

    curr_position = 0

    mom_df = pd.Series(mom_df[symbol], index=mom_df.index)
    prices_sma_ratio_df = pd.Series(prices_sma_ratio[symbol], index=prices_sma_ratio.index)
    bb_ratio_df = pd.Series(bb_ratio_df[symbol], index=bb_ratio_df.index)
    macd_signal = pd.Series(macd_info_df['MACDsign'], index=macd_info_df.index)
    macd_vals = pd.Series(macd_info_df['MACD'], index=macd_info_df.index)

    for i in range(0, prices.shape[0]):

        if i == prices.shape[0] - 1:
            break

        mom_today = mom_df[i]
        price_sma_ratio_today = prices_sma_ratio_df[i]
        bb_ratio_today = bb_ratio_df[i]
        macd_today =  macd_vals[i]
        macd_signal_today = macd_signal[i]

        # -1 Short, 0 Out, 1 Long
        signal = 0

        # Set signal based on indicators
        if mom_today < 0.0 and bb_ratio_today < 0.2 and (macd_today > macd_signal_today):
            signal = 1
        elif mom_today > 0.0 and bb_ratio_today > 0.8 and (macd_today < macd_signal_today):
            signal = -1


        # Adjust shares based on buy/sell signals.
        if signal == -1:
            if curr_position > -1000:
                new_amt = -1000 - curr_position
                curr_position = curr_position + new_amt
                df.iloc[i, df.columns.get_loc('Shares')] = new_amt
        elif signal == 1:
            if curr_position < 1000:
                new_amt = 1000 - curr_position
                curr_position = curr_position + new_amt
                df.iloc[i, df.columns.get_loc('Shares')] = new_amt

    return df


if __name__ == "__main__":
    """
    Call testpolicy and generate charts
    """
    symbol = "JPM"
    sdate = dt.datetime(2008, 1, 1)
    edate = dt.datetime(2009, 12, 31)
    outsample_sdate = dt.datetime(2010, 1, 1)
    outsample_edate = dt.datetime(2011, 12, 31)
    sv = 100000
    commission = 9.95
    impact = 0.005

    # CALL API FOR MANUAL INSAMPLE, OUTSAMPLE AND BENCHMARKS DATA
    df_manual_trades = testPolicy(symbol, sdate, edate, sv)
    df_manual_trades_out = testPolicy(symbol, outsample_sdate, outsample_edate, sv)

    df_benchmark = df_manual_trades.copy()
    df_benchmark['Shares'] = 0
    df_benchmark.Shares.iloc[0] = 1000

    df_outsample_benchmark = df_manual_trades_out.copy()
    df_outsample_benchmark['Shares'] = 0
    df_outsample_benchmark.Shares.iloc[0] = 1000

    # CONVERT TRADE DATA TO FORMAT ORDERBOOK FORMAT
    df_orders = marketsim.shares_to_orderbook(df_manual_trades, symbol)
    df_benchmark_orders = marketsim.shares_to_orderbook(df_benchmark, symbol)
    df_outsample_orders = marketsim.shares_to_orderbook(df_manual_trades_out, symbol)
    df_outsample_benchmark_orders = marketsim.shares_to_orderbook(df_outsample_benchmark, symbol)

    # COMPUTE PORTFOLIO VALUES, **COMMISSION/IMPACT ADDED HERE**
    portvals = marketsim.compute_portvals(df_orders, sv, commission, impact)
    portvals_benchmark = marketsim.compute_portvals(df_benchmark_orders, sv, commission, impact)
    portvals_outsample = marketsim.compute_portvals(df_outsample_orders, sv, commission, impact)
    portvals_outsample_benchmark = marketsim.compute_portvals(df_outsample_benchmark_orders, sv, commission, impact)

    # COMPUTE VARIOUS STATISTICS
    print "======= IN SAMPLE STATS =========="
    portvals_series = pd.Series(portvals['VALUE'], index=portvals.index)
    print marketsim.compute_stats(portvals_series)

    print "======= IN SAMPLE BENCHMARK =========="
    portvals_benchmark_series = pd.Series(portvals_benchmark['VALUE'], index=portvals_benchmark.index)
    print marketsim.compute_stats(portvals_benchmark_series)

    print "======= OUT SAMPLE STATS =========="
    portvals_outsample_series = pd.Series(portvals_outsample['VALUE'], index=portvals_outsample.index)
    print marketsim.compute_stats(portvals_outsample_series)

    print "======= OUT SAMPLE BENCHMARK =========="
    portvals_outsample_benchmark_series = pd.Series(portvals_outsample_benchmark['VALUE'], index=portvals_outsample_benchmark.index)
    print marketsim.compute_stats(portvals_outsample_benchmark_series)

    # NORMALIZE PRICES
    portvals_series = portvals_series / portvals_series[0]
    portvals_benchmark_series = portvals_benchmark_series / portvals_benchmark_series[0]
    portvals_outsample_series = portvals_outsample_series / portvals_outsample_series[0]
    portvals_outsample_benchmark_series = portvals_outsample_benchmark_series / portvals_outsample_benchmark_series[0]

    # PLOT GRAPHS

    # IN SAMPLE
    opt_plot = pd.concat([portvals_series, portvals_benchmark_series], axis=1)
    opt_plot.columns = ['JPM Using Indicators', 'JPM Benchmark']
    opt_plot.plot(grid=True, title='Manual Strategy vs Benchmark - In Sample', use_index=True, color=["r", "g"])
    plt.grid(True)
    plt.ylabel("Normed Price")
    plt.xlabel("Date")

    # Vertical lines: blue buy / black sell
    insample_long = df_manual_trades[df_manual_trades > 0]
    insample_long = insample_long.dropna()
    insample_short = df_manual_trades[df_manual_trades < 0]
    insample_short = insample_short.dropna()
    print insample_long.iloc[0]
    for index, row in insample_long.iterrows():
        plt.axvline(index, color='blue', linestyle='--', linewidth=0.8)
    for index, row in insample_short.iterrows():
        plt.axvline(index, color='black', linestyle='--', linewidth=0.8)

    filename = "ManualStrategyInsample.png"
    plt.savefig(filename, format='png', bbox_inches="tight")

    # OUTSAMPLE

    opt_plot = pd.concat([portvals_outsample_series, portvals_outsample_benchmark_series], axis=1)
    opt_plot.columns = ['JPM Using Indicators', 'JPM Benchmark']
    opt_plot.plot(grid=True, title='Manual Strategy vs Benchmark - Out sample', use_index=True, color=["r", "g"])
    plt.grid(True)
    plt.ylabel("Normed Price")
    plt.xlabel("Date")

    # Vertical lines: blue buy / black sell
    insample_long = df_manual_trades_out[df_manual_trades_out > 0]
    insample_long = insample_long.dropna()
    insample_short = df_manual_trades_out[df_manual_trades_out < 0]
    insample_short = insample_short.dropna()
    for index, row in insample_long.iterrows():
        plt.axvline(index, color='blue', linestyle='--', linewidth=0.8)
    for index, row in insample_short.iterrows():
        plt.axvline(index, color='black', linestyle='--', linewidth=0.8)

    filename = "ManualStrategyOutsample.png"
    plt.savefig(filename, format='png', bbox_inches="tight")

