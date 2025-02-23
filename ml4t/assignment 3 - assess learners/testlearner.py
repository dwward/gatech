""" 			  		 			 	 	 		 		 	  		   	  			  	
Test a learner.  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
"""

import argparse
import numpy as np
import math
import LinRegLearner as lrl
import BagLearner as bgl
import InsaneLearner as il
import DTLearner as dtl
import RTLearner as rtl
import util as util
import pandas as pd
import matplotlib.pyplot as plot
import time
import random
import os
import sys


def read_file(datafile):
    with util.get_learner_data_file(datafile) as f:
        datafile = np.genfromtxt(f, delimiter=',')
    return datafile


def lr_test():
    data = read_file("simple.csv")

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.addEvidence(trainX, trainY)  # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX)  # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0, 1]

    # evaluate out of sample
    predY = learner.query(testX)  # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0, 1]


def dt_test():
    data = read_file("excel.csv")
    data = data[1:-1]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    learner = dtl.DTLearner(verbose=True)
    learner.addEvidence(trainX, trainY)
    q = np.array([0.320, 0.780, 10.000]) # should return 6
    result = learner.query(q)
    print "Bag, queried with: 0.320, 0.780, 10.000 and result: ", result


def rt_test():
    data = read_file("excel.csv")
    data = data[1:-1]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    learner = rtl.RTLearner(verbose=True)
    learner.addEvidence(trainX, trainY)
    q = np.array([0.320, 0.780, 10.000]) # should return 6
    result = learner.query(q)
    print "Bag, queried with: 0.320, 0.780, 10.000 and result: ", result


def insane_test():
    data = read_file("simple.csv")
    data = data[1:-1]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    insane_learner = il.InsaneLearner()
    insane_learner.addEvidence(trainX, trainY)
    result = insane_learner.query(trainX)
    print "Insane queried: ", result


def bag_test():
    data = read_file("excel.csv")
    data = data[1:-1]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    bag_learner = bgl.BagLearner(dtl.DTLearner, verbose=True)
    bag_learner.addEvidence(trainX, trainY)
    q = np.array([0.320, 0.780, 10.000]) # should return 6
    result = bag_learner.query(q)
    print "Bag, queried with: 0.320, 0.780, 10.000 and result: ", result


def generate_report():

    # Read Data and strip columns
    data = read_file("Istanbul.csv")
    alldata = data[1:, 1:]
    datasize = alldata.shape[0]

    # Extract 60/40 split for train and test sets
    cutoff = int(datasize * 0.6)
    permutation = np.random.permutation(alldata.shape[0])
    col_permutation = np.random.permutation(alldata.shape[1] - 1)
    train_data = alldata[permutation[:cutoff], :]

    # trainX = train_data[:,:-1]
    trainX = train_data[:, col_permutation]
    trainY = train_data[:, -1]
    test_data = alldata[permutation[cutoff:], :]

    # testX = test_data[:,:-1]
    testX = test_data[:, col_permutation]
    testY = test_data[:, -1]
    msgs = []


    #
    # QUESTION 1: INCREASE LEAF SIZE AND OBSERVE RMSE
    #
    print "Report question 1 plots"

    rmse = []
    for leaf_sz in range(1, 125):
        learner = dtl.DTLearner(leaf_size=leaf_sz, verbose=False)
        learner.addEvidence(trainX, trainY)
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        in_rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])

        rmse.append([in_rmse, out_rmse])

    df = pd.DataFrame(rmse, columns=["In-Sample" , "Out-Sample"])
    df.plot()

    plot.title("Decision Tree Istanbul Set")
    plot.ylabel("RMSE")
    plot.xlabel("Leaf Size")
    # filename = "{}-{}_{}.png".format(sd.strftime("%m%Y"), ed.strftime("%m%Y"), "_".join(syms))
    plot.savefig("question1-leaf-vs-rmse.png", format='png', bbox_inches="tight")
    #plot.show()

    #
    # QUESTION 2: USE FIXED BAGS AND SEE IF LEAF SIZE HAS EFFECTS
    #
    print "Report question 2 plots"

    rmse = []
    numbags = 20
    for leaf_sz in range(1, 100):
        learner = bgl.BagLearner(dtl.DTLearner, kwargs={"leaf_size": leaf_sz}, bags=numbags, verbose=False)
        learner.addEvidence(trainX, trainY)
        # evaluate in sample
        predY = learner.query(trainX)  # get the predictions
        in_rmse = math.sqrt(((trainY - predY) ** 2).sum() / trainY.shape[0])

        # evaluate out of sample
        predY = learner.query(testX)  # get the predictions
        out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])

        rmse.append([in_rmse, out_rmse])

    df = pd.DataFrame(rmse, columns=["In-Sample", "Out-Sample"])
    df.plot()

    plot.title("Decision Tree Istanbul Set (bags={})".format(numbags))
    plot.ylabel("RMSE")
    plot.xlabel("Leaf Size")
    plot.savefig("question2-leaf-vs-20bags.png", format='png', bbox_inches="tight")
    #plot.show()

    #
    # QUESTION 3: COMPARE PROS CONS DT VS RT
    #

    print "Report question 3 plots"

    #
    # Question 3A: RMSE of DT vs RT
    #
    rmse = []
    for leaf_sz in range(1, 100):
        dtlearner = dtl.DTLearner(leaf_size=leaf_sz, verbose=False)
        rtlearner = rtl.RTLearner(leaf_size=leaf_sz, verbose=False)

        dtlearner.addEvidence(trainX, trainY)
        rtlearner.addEvidence(trainX, trainY)

        # evaluate out of sample
        predY = dtlearner.query(testX)  # get the predictions
        dt_out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])

        # evaluate out of sample
        predY = rtlearner.query(testX)  # get the predictions
        rt_out_rmse = math.sqrt(((testY - predY) ** 2).sum() / testY.shape[0])

        rmse.append([dt_out_rmse, rt_out_rmse])

    df = pd.DataFrame(rmse, columns=["DtLearner Out-Sample", "RtLearner Out-Sample"])
    df.plot()

    plot.title("Decision Tree Istanbul Set")
    plot.ylabel("RMSE")
    plot.xlabel("Leaf Size")
    # filename = "{}-{}_{}.png".format(sd.strftime("%m%Y"), ed.strftime("%m%Y"), "_".join(syms))
    plot.savefig("question3-dt-rt-rmse.png", format='png', bbox_inches="tight")

    # Question 3B: Time to query
    #
    # dt query | rt query
    mae = []

    for i in range(0, 200):
        dtlearner = dtl.DTLearner(leaf_size=i, verbose=False)
        rtlearner = rtl.RTLearner(leaf_size=i, verbose=False)

        dtlearner.addEvidence(trainX, trainY)
        rtlearner.addEvidence(trainX, trainY)

        # evaluate out of sample
        predY = dtlearner.query(testX)  # get the predictions
        dt_out_mae = (testY - predY).sum() / testY.shape[0]

        # evaluate out of sample
        predY = rtlearner.query(testX)  # get the predictions
        rt_out_mae = (testY - predY).sum() / testY.shape[0]

        mae.append([dt_out_mae, rt_out_mae])

    df = pd.DataFrame(mae, columns=["DtLearner Out-Sample", "RtLearner Out-Sample"])
    df.plot(df.index)
    plot.title("Decision Tree Istanbul Set - Mean Absolute Error vs Leaf Size")
    plot.ylabel("MAE")
    plot.xlabel("Leaf Size")
    # filename = "{}-{}_{}.png".format(sd.strftime("%m%Y"), ed.strftime("%m%Y"), "_".join(syms))
    plot.savefig("question3-dt-rt-mae.png", format='png', bbox_inches="tight")


    #
    # Question 3C: Build and Query comparison
    #
    # dt build | rt build
    build_times = []
    for leaf_sz in range(1, 50):
        dtlearner = dtl.DTLearner(leaf_size=leaf_sz, verbose=False)
        rtlearner = rtl.RTLearner(leaf_size=leaf_sz, verbose=False)

        start_time1 = time.time()
        dtlearner.addEvidence(trainX, trainY)
        elapsed_time1 = time.time() - start_time1

        start_time2 = time.time()
        rtlearner.addEvidence(trainX, trainY)
        elapsed_time2 = time.time() - start_time2

        build_times.append([elapsed_time1, elapsed_time2])

    df = pd.DataFrame(build_times, columns=["DtLearner Out-Sample", "RtLearner Out-Sample"])
    df.plot()
    plot.title("Decision Tree Istanbul Set - Build Tree Time")
    plot.ylabel("Build time seconds")
    plot.xlabel("Leaf Size")
    # filename = "{}-{}_{}.png".format(sd.strftime("%m%Y"), ed.strftime("%m%Y"), "_".join(syms))
    # plt.savefig(filename, format='png', bbox_inches="tight")
    plot.savefig("question3-dt-rt-build-times.png", format='png', bbox_inches="tight")

    print "Completed Report"




if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument('--filename', required=False, help="Provides optional filename to learner to try and use")
    parser.add_argument('--learner', required=False,
                        help="Learner to test: dt=decision tree, lr=linear regression, bag=bag learner, rt=random tree "
                             "insane=insane learner")
    args = parser.parse_args()
    filename_arg = args.filename
    learner_arg = args.learner
    print "learner_arg given: ", learner_arg
    if learner_arg == 'dt':
        dt_test()
    elif learner_arg == 'rt':
        rt_test()
    elif learner_arg == 'lr':
        lr_test()
    elif learner_arg == 'bag':
        bag_test()
    elif learner_arg == 'insane':
        insane_test()
    elif learner_arg == 'report' or 'None':
        generate_report()


