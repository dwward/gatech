"""MC1-P2: Optimize a portfolio. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# My imports
import scipy.optimize as spo
import matplotlib.dates as mdates


#
# Calculate daily portfolio values
#
def portfolio_value(allocs, prices, investment):
    # Normalized prices starting from 1.0 (shows percentage gain)
    normed = prices / prices.iloc[0]

    # Table of investment allocations by day
    alloced = normed * allocs

    # Table of position values by day
    pos_vals = alloced * investment

    # Value of portfolio each day
    port_val = pos_vals.sum(axis=1)

    return port_val


# ---------------------------------------------------------
# Daily returns based on allocations and investment amount
#
# Parameters: prices, allocs, investment
#   Pandas dataframe of stock prices, allocation amounts and a
#   starting investment amount.
# Returns:
#   Pandas dataframe of daily returns amounts indexed by date
#
def daily_returns(allocs, prices, investment):
    # Get daily value of portfolio
    port_val = portfolio_value(allocs, prices, investment)

    # Daily returns
    # * Important: The first value is always 0, we need to exclude.
    # Note: This will make the first row value NaN
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]

    return daily_rets


# ---------------------------------------------------------
# Sharpe Ratio Calculation Function
#
# Parameters:
#   prices: Dataframe of stock prices
#   allocs: Numpy array of allocation (must sum to 1.0)
#   k:      Sample rate
#   rfr:    Risk free return -- can use ZERO this day and age but can change.  Should
#           actually use table of values for real-life (LIBOR) but we use a constant for now
#   investment: Starting investment amount
#
# Returns:
#   Sharpe Ratio: _k_ * mean(_daily_rets_ - _daily_rf_) / std(_daily_rets_)
#       We negate sharpe ratio because we want maximum but spo.minimize finds minimum
#
def sr_func(allocs, prices, investment, k=252, rfr=0):
    daily_rets = daily_returns(allocs, prices, investment)
    sr = np.sqrt(k) * (daily_rets - rfr).mean() / daily_rets.std()
    return sr


# ---------------------------------------------------------
# Various portfolio statistics based on daily returns
#
def portfolio_stats(allocs, prices, investment):
    # Daily returns
    daily_rets = daily_returns(allocs, prices, investment)

    # Sharpe ratio
    sr = sr_func(allocs, prices, investment)

    # Cumulative return.  Today's price divided by first price
    portval = portfolio_value(allocs, prices, investment)
    cr = portval[-1] / portval[0] - 1

    # Average daily return.  Average of numbers.
    avg_daily_ret = daily_rets.mean()

    # Standard deviation of daily return
    std_daily_ret = daily_rets.std()

    return sr, cr, avg_daily_ret, std_daily_ret


# ---------------------------------------------------------
# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality 			  		 			 	 	 		 		 	  		   	  			  	
def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                       gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY

    #Fill missing data
    prices_all.fillna(method='ffill', inplace=True)
    prices_all.fillna(method='bfill', inplace=True)

    prices = prices_all[syms]  # only portfolio symbols 			  		 			 	 	 		 		 	  		   	  			  	
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later



    # ---------------------------------------
    # ADD CODE HERE to find the allocations
    # ---------------------------------------

    # investment amount
    investment = 1

    # guess allocations
    symbol_count = len(syms)
    guess_alloc = np.empty(symbol_count)
    guess_alloc.fill(1.0 / symbol_count)

    # allocation range
    bound_range = [(0, 1)] * symbol_count

    def neg_sharpe(*args):
        return sr_func(*args) * -1

    result = spo.minimize(neg_sharpe,
                          guess_alloc,
                          args=(prices, investment,),
                          method='SLSQP',
                          constraints=({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}),
                          bounds=bound_range,
                          options={'disp': True}
                          )
    allocs = result.x
    allocs = [round(elem, 2) for elem in allocs]  # rounding

    # ---------------------------------------
    # ADD CODE HERE to compute stats
    # cr = cumulative return
    # adr = average daily return
    # sddr = stddev daily return (volatility)
    # sr = sharpe ratio
    # ---------------------------------------
    sr, cr, adr, sddr = portfolio_stats(allocs, prices, investment)

    # ---------------------------------------
    # ADD CODE HERE to compute daily portfolio values
    # Compare daily portfolio value with SPY using a normalized plot
    # ---------------------------------------
    portval = portfolio_value(allocs, prices, investment)
    prices_SPY = prices_SPY / prices_SPY.ix[0]
    portval[0] = 1

    if gen_plot:
        # ---------------------------------------
        # ADD CODE HERE to plot here
        # ---------------------------------------
        df_temp = pd.concat([portval, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()

        # Format date labels
        xax = plt.gca().xaxis
        xax.set_major_formatter(mdates.DateFormatter("%b %Y"))

        # Tick marks on each month
        locator = mdates.MonthLocator()  # every month
        xax.set_major_locator(locator)

        plt.grid(True)
        plt.xticks(rotation='60')
        plt.title("Daily Portfolio Value and SPY")
        plt.ylabel("Price")
        plt.xlabel("Date")
        filename = "{}-{}_{}.png".format(sd.strftime("%m%Y"), ed.strftime("%m%Y"), "_".join(syms))

        plt.savefig(filename, format='png', bbox_inches="tight")

    # Return values:
    # -----------------
    # allocs: A 1-d Numpy ndarray of allocations to the stocks. All the allocations must be between 0.0 and 1.0 and they must sum to 1.0.
    # cr: Cumulative return
    # adr: Average daily return
    # sddr: Standard deviation of daily return
    # sr: Sharpe ratio
    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader 			  		 			 	 	 		 		 	  		   	  			  	
    # Do not assume that any variables defined here are available to your function/code 			  		 			 	 	 		 		 	  		   	  			  	
    # It is only here to help you set up and test your code 			  		 			 	 	 		 		 	  		   	  			  	

    # Define input parameters 			  		 			 	 	 		 		 	  		   	  			  	
    # Note that ALL of these values will be set to different values by 			  		 			 	 	 		 		 	  		   	  			  	
    # the autograder! 			  		 			 	 	 		 		 	  		   	  			  	

    # start_date = dt.datetime(2009, 1, 1)
    # end_date = dt.datetime(2010, 1, 1)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM', 'X', 'GLD', 'JPM']

    # Assess the portfolio 			  		 			 	 	 		 		 	  		   	  			  	
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, syms=symbols, gen_plot=True)

    # Print statistics 			  		 			 	 	 		 		 	  		   	  			  	
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader 			  		 			 	 	 		 		 	  		   	  			  	
    # Do not assume that it will be called 			  		 			 	 	 		 		 	  		   	  			  	
    test_code()
