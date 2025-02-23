'''

Experiment 1:

Student Name: Donald Ward (replace with your name)
GT User ID: dward45 (replace with your User ID)
GT ID: 903271210 (replace with your GT ID)
'''

import datetime as dt
import pandas as pd
import marketsimcode as marketsim
import numpy as np
import StrategyLearner as sl
import ManualStrategy as manualstrategy

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
    ql_impact = 0.00
    pv_impact = 0.00

    learner = sl.StrategyLearner(verbose=False, impact=ql_impact)  # Impact will vary
    learner.addEvidence(symbol=sym, sd=sdate, ed=edate, sv=sv)  # training phase

    # In-sample trades
    df_trades_in_ql = learner.testPolicy(symbol=sym, sd=sdate, ed=edate, sv=sv)  # testing phase
    df_trades_in_bench = learner.generate_benchmark_trades(df_trades_in_ql, sym)
    df_trades_manual_in = manualstrategy.testPolicy(sym, sdate, edate, sv)

    portvals_series = marketsim.trades2portvals(sym, df_trades_in_ql, sv, commission, pv_impact)
    portvals_benchmark_series = marketsim.trades2portvals(sym, df_trades_in_bench, sv, commission, pv_impact)
    pvals_manual_in = marketsim.trades2portvals(sym, df_trades_manual_in, sv, commission, pv_impact)

    # ---------------------------------------------
    # PLOT GRAPHS
    # ---------------------------------------------

    # IN SAMPLE
    opt_plot = pd.concat([portvals_series, portvals_benchmark_series, pvals_manual_in], axis=1)
    opt_plot.columns = [sym + ' Using Indicators', sym + ' Benchmark', sym + ' Manual']
    opt_plot.plot(grid=True, title='QLearner Strategy vs Manual Strategy - In Sample', use_index=True,
                  color=["r", "g", "y"])
    plt.grid(True)
    plt.ylabel("Normed Value")
    plt.xlabel("Date")

    filename = "experiment1_qlearner_vs_manual.png"
    plt.savefig(filename, format='png', bbox_inches="tight")
