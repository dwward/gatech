""" 			  		 			 	 	 		 		 	  		   	  			  	
A simple wrapper for linear regression.  (c) 2015 Tucker Balch 			  		 			 	 	 		 		 	  		   	  			  	
 			  		 			 	 	 		 		 	  		   	  			  	
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
import numpy as np


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.tree = np.array([])

    def author(self):
        return 'dward45'  # replace tb34 with your Georgia Tech username

    def addEvidence(self, data_x, data_y):
        """
        @summary: Add training data to learner
        @param data_x: X values of data to add
        @param data_y: the Y training values
        """

        # Attach Y data with X data s
        data_y = data_y.reshape((-1, 1))
        data = np.concatenate((data_x, data_y), axis=1)
        self.tree = self.build_tree(data)

    def query(self, features):
        """
        @summary: Query against a decision tree
        @param features: Numpy array with each row corresponding to a specific query.
        @returns Numpy array with the estimated values according to the saved model
        """
        # Convert 1d array to 2d, so we can iterate by row
        if features.ndim == 1:
            features = np.array([features])

        results = np.array([])

        try:
            for f in features:
                curr_row_idx = 0
                while True:

                    # Get the next node of the tree
                    curr_row = self.tree[int(curr_row_idx), :]
                    f_index = int(curr_row[0])
                    split_val = curr_row[1]
                    left_adv = curr_row[2]
                    right_adv = curr_row[3]

                    # Leaf found, append and move to next query
                    if f_index == -1:
                        results = np.append(results, split_val)
                        break

                    # Directions to the next node
                    query_val = f[f_index]
                    if query_val <= split_val:
                        curr_row_idx = curr_row_idx + left_adv
                    else:
                        curr_row_idx = curr_row_idx + right_adv

        except Exception as e:
            print e

        return results

    def build_tree(self, data):
        """
        @summary: Builds a decision tree given data for X and Y factors
        @param data: Numpy array with X factors and Y result data
        @returns Numpy array representing a decision-tree model
            Each leaf contains:
            X FACTOR | SPLITVAL | LEFT | RIGHT
            note: If X == -1, split value will contain the leaf value.
            otherwise, if X != -1, split value contains the discriminator.
        """

        def __all_data_y_same(tmp_data):
            y_col_bool = (tmp_data[:, -1] == tmp_data[0, -1])
            return np.all(y_col_bool)

        def __determine_split_feature(tmp_data):
            num_cols = tmp_data.shape[1]
            best_feature = (0, 0)  # index, correlation
            y_col = data[:, -1]

            # Determines highest correlation
            for c in range(0, num_cols - 1):
                x_col = data[:, c]
                corr = abs(np.corrcoef(x_col, y_col)[0, 1])
                if not corr:
                    corr = 0
                # Excel rounds to two places here
                # corr = np.round(corr, 2)

                if corr > best_feature[1]:
                    best_feature = (c, corr)

            # print "index: ", c, "corr: ", corr
            return best_feature[0]

        def __count_rows(arr):
            return 1 if arr.ndim == 1 else arr.shape[0]

        dx = data[:, :-1]
        dy = data[:, -1]

        # if there are leaf_size or fewer elements at the time of the recursive call,
        # the data should be aggregated into a leaf."
        if dx.shape[0] <= self.leaf_size:
            mean = dy.mean()
            return np.array([[-1, mean, -1, -1]])
        elif __all_data_y_same(data):
            return np.array([[-1, dy[0], -1, -1]])
        else:
            i = __determine_split_feature(data)
            split_val = np.median(dx[:, i])
            left_split = data[data[:, i] <= split_val]
            right_split = data[data[:, i] > split_val]

            if (right_split.size == 0 and left_split.size > 0) or (left_split.size == 0 and right_split.size > 0):
                mean = dy.mean()
                return np.array([[-1, mean, -1, -1]])

            lefttree = self.build_tree(left_split)
            righttree = self.build_tree(right_split)

            lefttree_rows = __count_rows(lefttree)
            root = [i, split_val, 1, lefttree_rows + 1]
            new_tree = np.vstack((root, lefttree, righttree))

            return new_tree


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
