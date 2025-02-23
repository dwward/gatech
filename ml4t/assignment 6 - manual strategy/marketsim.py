"""MC2-P1: Market simulator. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
from util import get_orders_data_file


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    '''
    :param orders_file: File or name of a file from which to read orders, and
    :param start_val: starting value of the portfolio (initial cash available)
    :param commission: fixed amount in dollars charged for each transaction (both entry and exit)
    :param impact: amount the price moves against the trader compared to the historical data at each transaction
    :return:
    '''

    def __read_orders_file(order_file_or_name):
        if isinstance(order_file_or_name, basestring):
            tmpfile = order_file_or_name
        else:
            tmpfile = order_file_or_name.name
        df_temp = pd.read_csv(tmpfile, index_col='Date',
                              parse_dates=True, usecols=['Date', 'Symbol', 'Order', 'Shares'], na_values=['nan'])
        df_temp.sort_index()
        return df_temp

    def __get_prices(symbols, sd, ed):
        # Read in adjusted closing prices for given symbols, date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data(symbols, dates)  # automatically adds SPY

        # Fill missing data
        prices_all.fillna(method='ffill', inplace=True)
        prices_all.fillna(method='bfill', inplace=True)

        prices = prices_all[symbols]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        return prices, prices_SPY

    def __calculate_holdings(orders_df, price_list, cash, commission, impact, sd, ed):
        shares_df = pd.DataFrame(0, columns=price_list.columns, index=price_list.index)
        shares_df['CASH'] = start_val

        for index, row in orders_df.iterrows():
            shares = row.Shares
            try:
                price = price_list.loc[index, row.Symbol]
            except KeyError:
                print "Warning: Tried to order {} when market closed ({})".format(row.Symbol, index)
                continue
            order_price = shares * price
            fees = commission + (impact * order_price)
            if row.Order == 'BUY':
                shares_df.loc[index:ed, 'CASH'] = shares_df.loc[index:ed, 'CASH'] - order_price - fees
                shares_df.loc[index:ed, row.Symbol] = shares_df.loc[index:ed, row.Symbol] + shares
            elif row.Order == 'SELL':
                shares_df.loc[index:ed, 'CASH'] = shares_df.loc[index:ed, 'CASH'] + order_price - fees
                shares_df.loc[index:ed, row.Symbol] = shares_df.loc[index:ed, row.Symbol] - shares

        return shares_df

    def __derive_portfolio_value(holdings, prices):
        portvals = pd.DataFrame(0, columns=['VALUE'], index=prices.index)
        prices['CASH'] = 1 # add a cash column to stock prices
        holdings_cash_value = holdings * prices
        portvals['VALUE'] = holdings_cash_value.sum(axis=1)
        return portvals

    orders_df = __read_orders_file(orders_file)
    # todo: sort orders
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    unique_symbols = orders_df['Symbol'].unique()

    prices, prices_SPY = __get_prices(unique_symbols.tolist(), start_date, end_date)
    holdings = __calculate_holdings(orders_df, prices, start_val, commission, impact, start_date, end_date)
    portvals = __derive_portfolio_value(holdings, prices)

    pd.options.display.float_format = '{:.2f}'.format
    print "\n", holdings

    return portvals


def compute_stats(portvals):
    start_date = portvals.index.min()
    end_date = portvals.index.max()

    # stats portfolio
    daily_rets = (portvals / portvals.shift(1)) - 1
    daily_rets = daily_rets[1:]
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * daily_rets.mean() / std_daily_ret
    cum_ret = portvals[-1] / portvals[0] - 1
    # stats SPY
    prices_SPY = get_data(['$SPX'], portvals.index)['$SPX']
    daily_rets_SPY = (prices_SPY / prices_SPY.shift(1)) - 1
    daily_rets_SPY = daily_rets_SPY[1:]
    avg_daily_ret_SPY = daily_rets_SPY.mean()
    std_daily_ret_SPY = daily_rets_SPY.std()
    sharpe_ratio_SPY = np.sqrt(252) * daily_rets_SPY.mean() / std_daily_ret_SPY
    cum_ret_SPY = prices_SPY[-1] / prices_SPY[0] - 1
    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


def test_code():
    # this is a helper function you can use to test your code 			  		 			 	 	 		 		 	  		   	  			  	
    # note that during autograding his function will not be called. 			  		 			 	 	 		 		 	  		   	  			  	
    # Define input parameters 			  		 			 	 	 		 		 	  		   	  			  	

    of = "./orders/orders-short.csv"
    sv = 1000000

    # Process orders 			  		 			 	 	 		 		 	  		   	  			  	
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats 			  		 			 	 	 		 		 	  		   	  			  	
    # Here we just fake the data. you should use your code from previous assignments. 			  		 			 	 	 		 		 	  		   	  			  	
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2, 0.01, 0.02, 1.5]
    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2, 0.01, 0.02, 1.5]

    compute_stats(portvals)


def author():
    return 'dward45'  # replace tb34 with your Georgia Tech username.


def test_orders_short():
    compute_portvals(orders_file="orders/orders-short.csv", start_val=1000000)


if __name__ == "__main__":
    test_code()
