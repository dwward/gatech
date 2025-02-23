'''

Experiment 2:

Student Name: Donald Ward (replace with your name)
GT User ID: dward45 (replace with your User ID)
GT ID: 903271210 (replace with your GT ID)
'''

import datetime as dt
import pandas as pd
import marketsimcode as marketsim
import numpy as np
import StrategyLearner as sl

from matplotlib import pyplot as plt


def author():
    return 'dward45'  # replace tb34 with your Georgia Tech username.


if __name__ == "__main__":
    np.random.seed(1)
    sym = "JPM"  # should be JPM
    sdate = dt.datetime(2008, 1, 1)
    edate = dt.datetime(2009, 12, 31)
    sv = 100000
    commission = 0.00
    ql_impact = [0.0, 0.001, 0.005]

    pvlist = []
    trades = []
    colors = ["r", "g", "y"]
    for impact in ql_impact:
        learner = sl.StrategyLearner(verbose=False, impact=impact)  # Impact will vary
        learner.addEvidence(symbol=sym, sd=sdate, ed=edate, sv=sv)  # training phase
        df_trades_in_ql = learner.testPolicy(symbol=sym, sd=sdate, ed=edate, sv=sv)  # testing phase
        portvals_series = marketsim.trades2portvals(sym, df_trades_in_ql, sv, commission, impact)
        pvlist.append(portvals_series)
        trades.append(df_trades_in_ql)

    # Plots first graph
    opt_plot = pd.concat(pvlist, axis=1)
    opt_plot.columns = [sym + ' Impact 0.000', sym + ' Impact 0.001', sym + ' Impact 0.005']
    opt_plot.plot(grid=True, title='QLearner JPM In-Sample w/ Impact', use_index=True,
                  color=["r", "g", "y"])
    plt.grid(True)
    plt.ylabel("Normed Value")
    plt.xlabel("Date")

    filename = "experiment2_qlearner_impact.png"
    plt.savefig(filename, format='png', bbox_inches="tight")

    # Plots second graph w/ trade frequency
    for i in range(0, 3):
        trd = trades[i]
        num_trades = trd.astype(bool).sum(axis=0)[0]

        # Chart 1
        ax1 = plt.subplot2grid((3, 1), (i, 0), rowspan=1)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Normed Values')
        # ax1.text(0.95, 0.30, "Trades: " + num_trades, transform=ax1.transAxes, fontsize=11,
        #          verticalalignment='bottom')

        opt_plot = pvlist[i]
        opt_plot.columns = ['Impact ' + str(ql_impact[i])]
        plt.grid(True)

        # Vertical lines: blue buy / black sell
        insample_long = trd[trd > 0]
        insample_long = insample_long.dropna()
        insample_short = trd[trd < 0]
        insample_short = insample_short.dropna()
        for index, row in insample_long.iterrows():
            plt.axvline(index, color='blue', linestyle='--', linewidth=0.8)
        for index, row in insample_short.iterrows():
            plt.axvline(index, color='black', linestyle='--', linewidth=0.8)

        opt_plot.plot(grid=True,
                      title='QLearner JPM Insample Trades w/ Impact: ' + str(ql_impact[i]) + "   [" + str(
                          num_trades) + " trades]",
                      use_index=True, ax=ax1,
                      color=colors[i], sharex=True)

    filename = "experiment2_qlearner_impact_tradefreq.png"
    plt.tight_layout()
    plt.savefig(filename, format='png', bbox_inches="tight")
