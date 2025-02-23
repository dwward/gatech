"""Assess a betting strategy. 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return 'dward45'  # replace tb34 with your Georgia Tech username.


def gtid():
    return 903271210  # replace with your GT ID number


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def test_code():
    # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once
    win_prob = bet_on_black_probability()
    print "Test Spin: ", get_spin_result(win_prob)  # test the roulette spin

    graph1()
    graph2()
    graph3()
    graph4()
    graph5()

    print "End Program"


def graph1():
    # add your code here to implement the experiments
    episodes = 10
    win_prob = bet_on_black_probability()
    spins = 1000
    target = 80

    # 10 EPISODES, PLOTTED
    records = np.empty((0, 1001))
    for i in xrange(0, episodes):
        spins_left, episode_winnings, record = gamble(spins, win_prob, target)
        records = np.vstack([records, record])

    df = pd.DataFrame()  # empty
    for j in xrange(0, 10):
        rec = records[j]
        df_tmp = pd.DataFrame({'Episode {}'.format(j + 1): rec})
        df = pd.concat([df, df_tmp], axis=1)
    df.plot()

    plt.title("Experiment 1 - Figure 1")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spin')
    plt.ylabel('Winnings')
    plt.savefig("figure1.png")



def graph2():
    # add your code here to implement the experiments
    episodes = 1000
    win_prob = bet_on_black_probability()
    spins = 1000
    target = 80

    records = np.empty((0, 1001))
    for i in xrange(0, episodes):
        spins_left, episode_winnings, record = gamble(spins, win_prob, target)
        records = np.vstack([records, record])

    # MEAN WITH STANDARD DEVIATION
    meany = np.zeros(1001)
    deviant_plus = np.zeros(1001)
    deviant_minus = np.zeros(1001)
    for x in range(0, 1001):
        curr_mean = records[0:, x].mean()
        meany[x] = curr_mean
        curr_sd = records[0:, x].std()
        deviant_plus[x] = curr_mean + curr_sd
        deviant_minus[x] = curr_mean - curr_sd

    df = pd.DataFrame()  # empty
    df_tmp = pd.DataFrame({'mean': meany})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std+': deviant_plus})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std+': deviant_minus})
    df = pd.concat([df, df_tmp], axis=1)
    df.plot()

    plt.title("Experiment 1 - Figure 2")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spin')
    plt.ylabel('Winnings')
    plt.savefig("figure2.png")


def graph3():
    # add your code here to implement the experiments
    episodes = 1000
    win_prob = bet_on_black_probability()
    spins = 1000
    target = 80

    records = np.empty((0, 1001))
    for i in xrange(0, episodes):
        spins_left, episode_winnings, record = gamble(spins, win_prob, target)
        records = np.vstack([records, record])

    # MEDIAN WITH STANDARD DEVIATION
    mead = np.zeros(1001)
    deviant_plus = np.zeros(1001)
    deviant_minus = np.zeros(1001)
    for x in range(0, 1001):
        curr_mead = np.median(records[0:, x])
        mead[x] = curr_mead
        curr_sd = np.std(records[0:, x])
        deviant_plus[x] = curr_mead + curr_sd
        deviant_minus[x] = curr_mead - curr_sd

    df = pd.DataFrame()  # empty
    df_tmp = pd.DataFrame({'median': mead})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std+': deviant_plus})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std-': deviant_minus})
    df = pd.concat([df, df_tmp], axis=1)
    df.plot()

    plt.title("Experiment 1 - Figure 3")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spin')
    plt.ylabel('Winnings')
    plt.savefig("figure3.png")



def graph4():
    # add your code here to implement the experiments
    episodes = 1000
    win_prob = bet_on_black_probability()
    spins = 1000
    target = 80
    bankroll = 256

    target_reached = 0
    average_earnings = 0

    records = np.empty((0, 1001))
    for i in xrange(0, episodes):
        spins_left, episode_winnings, record = gamble(spins, win_prob, target, bankroll)
        records = np.vstack([records, record])
        if episode_winnings >= 80:
            target_reached = target_reached + 1
        average_earnings = average_earnings + episode_winnings

    print "Probability of hitting target: ", target_reached
    print "Average earning: ", average_earnings / 1000

    # MEAN WITH STANDARD DEVIATION
    meany = np.zeros(1001)
    deviant_plus = np.zeros(1001)
    deviant_minus = np.zeros(1001)
    for x in range(0, 1001):
        curr_mean = records[0:, x].mean()
        meany[x] = curr_mean
        curr_sd = records[0:, x].std()
        deviant_plus[x] = curr_mean + curr_sd
        deviant_minus[x] = curr_mean - curr_sd

    df = pd.DataFrame()  # empty
    df_tmp = pd.DataFrame({'mean': meany})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std+': deviant_plus})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std-': deviant_minus})
    df = pd.concat([df, df_tmp], axis=1)
    df.plot()

    plt.title("Experiment 2 - Figure 1")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spin')
    plt.ylabel('Winnings')
    plt.savefig("figure4.png")


def graph5():
    # add your code here to implement the experiments
    episodes = 1000
    win_prob = bet_on_black_probability()
    spins = 1000
    target = 80
    bankroll = 256

    records = np.empty((0, 1001))
    for i in xrange(0, episodes):
        spins_left, episode_winnings, record = gamble(spins, win_prob, target, bankroll)
        records = np.vstack([records, record])

    # MEDIAN WITH STANDARD DEVIATION
    mead = np.zeros(1001)
    deviant_plus = np.zeros(1001)
    deviant_minus = np.zeros(1001)
    for x in range(0, 1001):
        curr_mead = np.median(records[0:, x])
        mead[x] = curr_mead
        curr_sd = np.std(records[0:, x])
        deviant_plus[x] = curr_mead + curr_sd
        deviant_minus[x] = curr_mead - curr_sd

    df = pd.DataFrame()  # empty
    df_tmp = pd.DataFrame({'median': mead})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std+': deviant_plus})
    df = pd.concat([df, df_tmp], axis=1)
    df_tmp = pd.DataFrame({'std-': deviant_minus})
    df = pd.concat([df, df_tmp], axis=1)
    df.plot()

    plt.title("Experiment 2 - Figure 2")
    plt.xlim(0, 300)
    plt.ylim(-256, 100)
    plt.xlabel('Spin')
    plt.ylabel('Winnings')
    plt.savefig("figure5.png")


def bet_on_black_probability():
    # 38 total slots, two occupied by zeros.  Remaining 36 split between black and red.
    return 18.0 / 38.0


def gamble(spins, win_probability, target_winnings, bankroll=9999999):
    episode_winnings = 0
    episode_results = np.zeros(spins + 1)  # Size+1, because we reserve zero index
    current_spin = 0

    # Keep gambling until we hit our target
    while (-1 * bankroll) <= episode_winnings < target_winnings and not current_spin > spins:
        won = False
        bet_amount = 1

        # Keep doubling bet until we win
        while not won:
            current_spin = current_spin + 1

            # Ran out of spins, stop wagering
            if current_spin > spins:
                break

            # If bankroll limit has not been hit, spin
            have_enough_to_bet = (episode_winnings - bet_amount) >= (-1 * bankroll)
            if have_enough_to_bet:
                won = get_spin_result(win_probability)

                # If we win
                if won:
                    episode_winnings = episode_winnings + bet_amount
                    # Special case, if target winnings are hit, set rest of array values
                    if episode_winnings >= target_winnings:
                        episode_results[current_spin:] = episode_winnings
                        break
                else:
                    episode_winnings = episode_winnings - bet_amount

                    # We lost!  Try to double our bet....
                    # Double bet if does not exceed bankroll, otherwise keep same bet
                    if episode_winnings - (bet_amount * 2) >= (-1 * bankroll):
                        bet_amount = bet_amount * 2
                    else:
                        bet_amount = episode_winnings - (-1 * bankroll)

            # track winnings
            episode_results[current_spin] = episode_winnings

    return spins - current_spin, episode_winnings, episode_results


if __name__ == "__main__":
    test_code()
