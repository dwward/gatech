""" 			  		 			 	 	 		 		 	  		   	  			  	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Copyright 2018, Georgia Institute of Technology (Georgia Tech) 			  		 			 	 	 		 		 	  		   	  			  	
Atlanta, Georgia 30332 			  		 			 	 	 		 		 	  		   	  			  	
All Rights Reserved 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Template code for CS 4646/7646 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Georgia Tech asserts copyright ownership of this template and all derivative 			  		 			 	 	 		 		 	  		   	  			  	
works, including solutions to the projects assigned in this course. Students 			  		 			 	 	 		 		 	  		   	  			  	
and other users of this template code are advised not to share it with others 			  		 			 	 	 		 		 	  		   	  			  	
or to make it available on publicly viewable websites including repositories 			  		 			 	 	 		 		 	  		   	  			  	
such as github and gitlab.  This copyright statement should not be removed 			  		 			 	 	 		 		 	  		   	  			  	
or edited. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
We do grant permission to share solutions privately with non-students such 			  		 			 	 	 		 		 	  		   	  			  	
as potential employers. However, sharing with other current or future 			  		 			 	 	 		 		 	  		   	  			  	
students of CS 7646 is prohibited and subject to being investigated as a 			  		 			 	 	 		 		 	  		   	  			  	
GT honor code violation. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
-----do not edit anything above this line--- 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
Student Name: Donald Ward (replace with your name)
GT User ID: dward45 (replace with your User ID)
GT ID: 903271210 (replace with your GT ID)
"""

import datetime as dt
import time
import pandas as pd
import marketsimcode as marketsim
import QLearner as ql
import numpy as np

from collections import OrderedDict
from indicators import bb, mom, macd
# from matplotlib import pyplot as plt


class StrategyLearner(object):

    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact  # assessed on daily
        self.qlearner = None
        self.default_bins = 9  #
        self.num_states = 6562
        self.num_actions = 3
        self.q_alpha = 0.5  # default 0.2
        self.q_gamma = 0.7  # default 0.9
        self.q_rar = 0.5  # default 0.5
        self.q_radr = 0.99  # default 0.99
        self.indicators = OrderedDict()
        self.indicators_rolling_days = 20
        self.indicators_momentum_days = 14
        self.indicators_adjusted_days_back = 60
        self.position_short = 0
        self.position_none = 1
        self.position_long = 2
        self.epochs = 150
        self.random_epochs = 60
        self.dyna = 0  # when bins are large, dyna blows up memory
        self.max_holding = 1000
        self.converge_seconds = 22

    def author(self):
        return 'dward45'  # replace tb34 with your Georgia Tech username.

    def add_indicator(self, name, df, thresh=None):
        if thresh is None:
            thresh = self.calc_thresholds(df)
        self.indicators[name] = {
            "df": df,
            "threshold": thresh
        }

    def calc_thresholds(self, df):
        df_tmp = df.sort_values()
        pt = pd.qcut(df_tmp, q=self.default_bins - 1, retbins=True)
        return pt[1]

    def get_state(self, date):
        indicators = self.indicators
        state = 0

        for i, (label, value) in enumerate(indicators.iteritems()):
            i_state = np.digitize(value["df"].loc[date], value["threshold"]) - 1  # digitize starts at 1
            state = state + i_state * self.default_bins ** i
        return state

    def calc_indicators(self, symbol, sd, ed, prices):
        try:
            # add your code to do learning here
            rolling = self.indicators_rolling_days
            #adj_sd = sd - dt.timedelta(self.indicators_adjusted_days_back)  # to get rolling data

            # Get BB Indicator
            bb_mean, bb_upper, bb_lower = bb(prices, rolling)
            bb_ratio_df = (prices - bb_lower) / (bb_upper - bb_lower)
            bb_ratio_df = bb_ratio_df[sd:ed]

            # Get MOM Indicator
            mom_days = self.indicators_momentum_days
            mom_df = mom(prices, mom_days)
            mom_df = mom_df[sd:ed]

            # Get MACD Indicator
            macd_info_df = macd(prices)
            macd_info_df = macd_info_df[sd:ed]

            # Strip off earlier dates used for rolling data
            # prices = prices[sd:ed]
            mom_df = pd.Series(mom_df[symbol], index=mom_df.index)
            bb_ratio_df = pd.Series(bb_ratio_df[symbol], index=bb_ratio_df.index)
            macd_signal = pd.Series(macd_info_df['MACDsign'], index=macd_info_df.index)
            macd_vals = pd.Series(macd_info_df['MACD'], index=macd_info_df.index)
            macd_cross = macd_vals - macd_signal

            # macdThresh = pd.qcut([macd_cross.min()-1, 0, macd_cross.max()+1], 2, retbins=True)
            # self.add_indicator("MACDcross", macd_cross, 2, macdThresh[1])
            # self.num_states=200
            # self.add_indicator("MACDcross", macd_cross)
            self.add_indicator("MACD", macd_vals)
            self.add_indicator("MACDsign", macd_signal)
            self.add_indicator("MOM", mom_df)
            self.add_indicator("BBP", bb_ratio_df)
        except ValueError:
            print "Error adding indicators"
            
        return self.indicators

    def converged(self, rounds, start_time, loss_trend):

        allowable_losses_trend = -20
        if loss_trend < allowable_losses_trend:
            if self.verbose:
                print "Consecutive losses exceeded: ", allowable_losses_trend
            return True
        if (time.time() - start_time) > self.converge_seconds:
            if self.verbose:
                print "Time reached max for converge: ", self.converge_seconds
            return True
        if rounds == self.epochs:
            if self.verbose:
                print self.epochs, " run in ", time.time() - start_time, " seconds"
            return True

    def addEvidence(self, symbol="AAPL", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):

        prices, prices_SPY = marketsim.get_data_by_column([symbol], sd, ed, marketsim.DATACOL_ADJUSTED_CLOSE)
        self.calc_indicators(symbol, sd, ed, prices)
        self.qlearner = ql.QLearner(num_states=self.num_states, num_actions=self.num_actions, dyna=self.dyna,
                                    alpha=self.q_alpha,
                                    gamma=self.q_gamma, rar=self.q_rar, radr=self.q_radr)

        rounds = 0
        start_time = time.time()
        loss_trend = 0

        # time, position, rounds
        if self.verbose:
            print "\nRunning Epochs:"
        while not self.converged(rounds, start_time, loss_trend):
            if self.verbose:
                print rounds + 1, 'out of', self.epochs, '\r',

            if rounds < self.random_epochs:
                self.qlearner.rar = 1

            state = self.get_state(prices.index[0])
            action = self.qlearner.querysetstate(state)
            last_action = None

            for day in prices.index[1:]:

                # Formula (share*price)*impact
                # if moving from short or long, assess impact
                current_price = prices[symbol].loc[day]
                previous_price = prices[symbol].iloc[prices.index.get_loc(day) - 1]

                # If we closed a position, then subtract impact
                impact_day = 0
                if self.impact > 0 and last_action != action:
                    if last_action == self.position_short or last_action == self.position_long:
                        impact_day = (current_price * self.impact)
                        # current_price = current_price - impact_day

                one_day_return = (current_price / previous_price) - 1

                if action == self.position_short:  # 0
                    r = -1 * (one_day_return - impact_day)
                elif action == self.position_none:  # 1
                    r = 0
                elif action == self.position_long:  # 2
                    r = (one_day_return - impact_day)

                last_action = action
                state = self.get_state(day)
                action = self.qlearner.query(state, r)

            rounds = rounds + 1

        self.qlearner.rar = 0

    def map_ql_action_to_signal(self, ql_action):
        if ql_action == self.position_short:
            return -1
        elif ql_action == self.position_none:
            return 0
        elif ql_action == self.position_long:
            return 1

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=10000):

        if not self.qlearner:
            raise Exception("Warning: Testing policy with uninitialized QLearner.  Run addEvidence first")

        start_time = time.time()

        prices, prices_SPY = marketsim.get_data_by_column([symbol], sd, ed, marketsim.DATACOL_ADJUSTED_CLOSE)
        prices = prices[sd:ed]
        self.calc_indicators(symbol, sd, ed, prices)
        df = pd.DataFrame(0, index=prices.index, columns=['Shares'])

        curr_position = 0
        for i in range(0, prices.shape[0]):

            # last day
            if i == prices.shape[0] - 1:
                break

            state = self.get_state(prices.index[i])
            action = self.qlearner.querysetstate(state)
            signal = self.map_ql_action_to_signal(action)  # convert qlearner actions to buy/sell signals

            # Adjust shares based on buy/sell signals.
            if signal == -1:
                if curr_position > -1 * self.max_holding:
                    new_amt = -1 * self.max_holding - curr_position
                    curr_position = curr_position + new_amt
                    df.iloc[i, df.columns.get_loc('Shares')] = new_amt
            elif signal == 1:
                if curr_position < self.max_holding:
                    new_amt = self.max_holding - curr_position
                    curr_position = curr_position + new_amt
                    df.iloc[i, df.columns.get_loc('Shares')] = new_amt

        end_time = time.time() - start_time
        if self.verbose:
            print "Test policy executed and returned in ", end_time, " seconds"

        return df

    def generate_benchmark_trades(self, reference_df, sym):
        tmp_df = reference_df.copy()
        tmp_df['Shares'] = 0
        tmp_df.Shares.iloc[0] = self.max_holding
        trades = marketsim.shares_to_orderbook(tmp_df, sym)
        return trades


if __name__ == "__main__":
    pass


    # """
    # Call testpolicy and generate charts
    # """
    #
    # sym = "JPM"  # should be JPM
    # sdate = dt.datetime(2008, 1, 1)
    # edate = dt.datetime(2009, 12, 31)
    # outsample_sdate = dt.datetime(2010, 1, 1)
    # outsample_edate = dt.datetime(2011, 12, 31)
    # sv = 100000
    # commission = 0.00
    # ql_impact = 0.00
    # pv_impact = 0.00
    #
    # learner = StrategyLearner(verbose=True, impact=ql_impact)  # Impact will vary
    # learner.addEvidence(symbol=sym, sd=sdate, ed=edate, sv=sv)  # training phase
    #
    # # In-sample trades
    # df_trades_in_ql = learner.testPolicy(symbol=sym, sd=sdate, ed=edate, sv=sv)  # testing phase
    # df_trades_in_bench = learner.generate_benchmark_trades(df_trades_in_ql, sym)
    #
    # # Out-sample trades
    # df_trades_out_ql = learner.testPolicy(symbol=sym, sd=outsample_sdate, ed=outsample_edate, sv=sv)  # testing phase
    # df_trades_out_bench = learner.generate_benchmark_trades(df_trades_out_ql, sym)
    #
    # # Manual strategy trades
    # df_trades_manual_in = manualstrategy.testPolicy(sym, sdate, edate, sv)
    # df_trades_manual_out = manualstrategy.testPolicy(sym, outsample_sdate, outsample_edate, sv)
    #
    # print ""
    # print "======= IN SAMPLE PORTFOLIO =========="
    # portvals_series = marketsim.trades2portvals(sym, df_trades_in_ql, sv, commission, pv_impact)
    #
    # print "======= IN SAMPLE PORTFOLIO BENCHMARK =========="
    # portvals_benchmark_series = marketsim.trades2portvals(sym, df_trades_in_bench, sv, commission, pv_impact)
    #
    # print "======= OUT SAMPLE PORTFOLIO =========="
    # portvals_outsample_series = marketsim.trades2portvals(sym, df_trades_out_ql, sv, commission, pv_impact)
    #
    # print "======= OUT SAMPLE PORTFOLIO BENCHMARK =========="
    # portvals_outsample_benchmark_series = marketsim.trades2portvals(sym, df_trades_out_bench, sv, commission, pv_impact)
    #
    # print "======= IN SAMPLE PORTFOLIO BENCHMARK =========="
    # pvals_manual_in = marketsim.trades2portvals(sym, df_trades_manual_in, sv, commission, pv_impact)
    #
    # print "======= OUT SAMPLE PORTFOLIO BENCHMARK =========="
    # pvals_manual_out = marketsim.trades2portvals(sym, df_trades_manual_out, sv, commission, pv_impact)
    #
    # # ---------------------------------------------
    # # PLOT GRAPHS
    # # ---------------------------------------------
    #
    # # IN SAMPLE
    # opt_plot = pd.concat([portvals_series, portvals_benchmark_series, pvals_manual_in], axis=1)
    # opt_plot.columns = [sym + ' Using Indicators', sym + ' Benchmark', sym + ' Manual']
    # opt_plot.plot(grid=True, title='QLearner Strategy vs Benchmark - In Sample', use_index=True, color=["r", "g", "y"])
    # plt.grid(True)
    # plt.ylabel("Normed Price")
    # plt.xlabel("Date")
    #
    # # Vertical lines: blue buy / black sell
    # insample_long = df_trades_in_ql[df_trades_in_ql > 0]
    # insample_long = insample_long.dropna()
    # insample_short = df_trades_in_ql[df_trades_in_ql < 0]
    # insample_short = insample_short.dropna()
    #
    # for index, row in insample_long.iterrows():
    #     plt.axvline(index, color='blue', linestyle='--', linewidth=0.8)
    # for index, row in insample_short.iterrows():
    #     plt.axvline(index, color='black', linestyle='--', linewidth=0.8)
    #
    # filename = "StrategyLearnerInsample.png"
    # #plt.savefig(filename, format='png', bbox_inches="tight")
    # plt.show()
    #
    # # OUTSAMPLE
    # opt_plot = pd.concat([portvals_outsample_series, portvals_outsample_benchmark_series, pvals_manual_out], axis=1)
    # opt_plot.columns = [sym + ' Using Indicators', sym + ' Benchmark', sym + " Manual"]
    # opt_plot.plot(grid=True, title='QLearner Strategy vs Benchmark - Out sample', use_index=True, color=["r", "g", "y"])
    # plt.grid(True)
    # plt.ylabel("Normed Price")
    # plt.xlabel("Date")
    #
    # # Vertical lines: blue buy / black sell
    # insample_long = df_trades_out_ql[df_trades_out_ql > 0]
    # insample_long = insample_long.dropna()
    # insample_short = df_trades_out_ql[df_trades_out_ql < 0]
    # insample_short = insample_short.dropna()
    # for index, row in insample_long.iterrows():
    #     plt.axvline(index, color='blue', linestyle='--', linewidth=0.8)
    # for index, row in insample_short.iterrows():
    #     plt.axvline(index, color='black', linestyle='--', linewidth=0.8)
    #
    # filename = "StrategyLearnerOutsample.png"
    # #plt.savefig(filename, format='png', bbox_inches="tight")
    # plt.show()
    #
    # # print "One does not simply think up a strategy"
