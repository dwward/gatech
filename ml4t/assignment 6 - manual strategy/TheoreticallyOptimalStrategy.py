'''Code implementing a TheoreticallyOptimalStrategy object (details below). It should
implement testPolicy() which returns a trades data frame (see below). The main part
of this code should call marketsimcode as necessary to generate the plots used in
the report.
'''
import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import marketsimcode as marketsim
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
    one_day = dt.timedelta(1)
    prices, prices_SPY = marketsim.get_data_by_column([symbol], sd, (ed + one_day), marketsim.DATACOL_ADJUSTED_CLOSE)
    df = pd.DataFrame(0, index=prices.index, columns=['Shares'])

    curr_position = 0
    for i in range(0, prices.shape[0]):

        # Can't peek at tomorrow's price to just quit
        if i == prices.shape[0]-1:
            break

        today_price = prices.iloc[i, 0]
        tomorrow_price = prices.iloc[i+1, 0]

        if today_price < tomorrow_price and curr_position < 1000:
            new_amt = 1000 - curr_position
            curr_position = curr_position + new_amt
            df.iloc[i, df.columns.get_loc('Shares')] = new_amt
        elif today_price > tomorrow_price and curr_position > -1000:
            new_amt = -1000 - curr_position
            curr_position = curr_position + new_amt
            df.iloc[i, df.columns.get_loc('Shares')] = new_amt

    return df


def generate_optimal_charts(optimal_df, benchmark_df):
    pass


if __name__ == "__main__":
    """
    Call testpolicy and generate charts
    """
    symbol = "JPM"
    sdate = dt.datetime(2008, 1, 1)
    edate = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0
    impact = 0

    df_optimal_trades = testPolicy(symbol, sdate, edate, sv)
    df_benchmark = df_optimal_trades.copy()
    df_benchmark['Shares'] = 0
    df_benchmark.Shares.iloc[0] = 1000

    df_orders = marketsim.shares_to_orderbook(df_optimal_trades, symbol)
    df_benchmark_orders = marketsim.shares_to_orderbook(df_benchmark, symbol)
    portvals = marketsim.compute_portvals(df_orders, sv, commission, impact)
    portvals_benchmark = marketsim.compute_portvals(df_benchmark_orders, sv, commission, impact)

    print "======= JPM OPTIMAL STATISTICS =========="
    portvals_series = pd.Series(portvals['VALUE'], index=portvals.index)
    print marketsim.compute_stats(portvals_series)

    print "======= JPM BENCHMARK STATISTICS =========="
    portvals_benchmark_series = pd.Series(portvals_benchmark['VALUE'], index=portvals_benchmark.index)
    print marketsim.compute_stats(portvals_benchmark_series)

    #norm prices
    portvals_series = portvals_series / portvals_series[0]
    portvals_benchmark_series = portvals_benchmark_series / portvals_benchmark_series[0]

    opt_plot = pd.concat([portvals_series, portvals_benchmark_series], axis=1)
    opt_plot.columns = ['JPM Optimal', 'JPM Benchmark']
    opt_plot.plot(grid=True, title='JPM Optimal', use_index=True, color=["r", "g"])
    # plt.xticks(rotation='60')
    plt.grid(True)
    plt.title("Theoretically Optimal Strategy vs Benchmark")
    plt.ylabel("Normed Price")
    plt.xlabel("Date")
    filename = "TheoreticallyOptimal.png"

    plt.savefig(filename, format='png', bbox_inches="tight")




