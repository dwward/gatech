# from typing import Union, Iterable

import numpy as np
from collections import Counter
import time


# from numpy.core._multiarray_umath import ndarray


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            # print("Return label:", self.class_label)
            return self.class_label
        elif self.decision_function(feature):
            # print("left")
            return self.left.decide(feature)
        else:
            # print("right")
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if (class_index == -1):
        classes = out[:, class_index]
        features = out[:, :class_index]
        return features, classes
    elif (class_index == 0):
        classes = out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def get_true_node():
    return DecisionNode(None, None, None, 1)


def get_false_node():
    return DecisionNode(None, None, None, 0)


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    """    
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        left (DecisionNode): left child node.
        right (DecisionNode): right child node.
        decision_function (func): function to decide left or right node.
        class_label (int): label for leaf node. Default is None.
    """

    # Node declaration
    decision_tree_root = DecisionNode(None, None, lambda f: f[0] == 1)
    node_right_a3 = DecisionNode(None, None, lambda f: f[2] == 1)
    node_right_left_a4 = DecisionNode(None, None, lambda f: f[3] == 1)
    node_right_right_a4 = DecisionNode(None, None, lambda f: f[3] == 1)

    # Wiring
    # -- left side
    decision_tree_root.left = get_true_node()
    decision_tree_root.right = node_right_a3

    # -- right side
    node_right_a3.left = node_right_left_a4
    node_right_a3.right = node_right_right_a4
    node_right_left_a4.left = get_true_node()
    node_right_left_a4.right = get_false_node()
    node_right_right_a4.left = get_false_node()
    node_right_right_a4.right = get_true_node()

    return decision_tree_root


#                  v---- [actual] -------v
# --------------------------------------------
#                | Has       | Does not have
# --------------------------------------------
# Has            | True pos  | False neg
# --------------------------------------------
# Does not Have  | False pos | True neg
# --------------------------------------------

def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative], [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """
    cm = [[0, 0], [0, 0]]

    for i in range(0, len(classifier_output)):
        if classifier_output[i] == 1 and true_labels[i] == 1:
            cm[0][0] += 1  # true pos
        elif classifier_output[i] == 1 and true_labels[i] == 0:
            cm[1][0] += 1  # false pos
        elif classifier_output[i] == 0 and true_labels[i] == 0:
            cm[1][1] += 1  # true neg
        elif classifier_output[i] == 0 and true_labels[i] == 1:
            cm[0][1] += 1  # false neg

    return cm


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    # cm.tp = cm[0][0]
    # cm.fp = cm[1][0]
    # cm.tn = cm[1][1]
    # cm.fn = cm[0][1]

    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] + cm[1][0])


def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    # cm.tp = cm[0][0]
    # cm.fp = cm[1][0]
    # cm.tn = cm[1][1]
    # cm.fn = cm[0][1]

    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] + cm[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    # cm.tp = cm[0][0]
    # cm.fp = cm[1][0]
    # cm.tn = cm[1][1]
    # cm.fn = cm[0][1]

    cm = confusion_matrix(classifier_output, true_labels)
    return (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    # For J classes, let p_i be the fraction of items labeled with class i
    # 1 minus Summation((p_i)**2) where i is all labels
    # I'm a genie in a bottle baby.....
    num_classes = len(class_vector)
    label_cts = Counter(class_vector)
    gini = 1
    for l in label_cts.most_common():
        gini = gini - (l[1] / num_classes) ** 2
    return gini


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    # information gain:  gini imp of parent minus gini imp of child nodes

    g_prev = gini_impurity(previous_classes)  # parent
    # g_curr = 0  # children sum
    num_lists = len(current_classes)
    total_sum = 0
    weights = []
    imps = []
    for n in current_classes:
        curr_list_len = len(n)  # keep for weighted sum
        weights.append(curr_list_len)
        total_sum = total_sum + curr_list_len  # use np func instead?
        imp = gini_impurity(n)
        imps.append(imp)
        # g_curr = g_curr + imp
    weights = np.asarray(weights)
    imps = np.asarray(imps)

    weights = weights / total_sum
    imps = imps * weights
    score = g_prev - np.sum(imps)
    return score


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        return self.tree_build(features, classes, depth)

    # # Experiment: Find split on MEDIAN
    # def find_split(self, table, col):
    #     table = table[table[:, col].argsort()]  # Sort by column
    #     median_index = len(table[:, col]) // 2
    #     median = table[median_index, col]
    #
    #     # Calculate some nearby gini values
    #     # -------------------------------
    #     """
    #     band_width = 0.045
    #     alen = len(table[:, col])
    #     num_indices = int(alen * band_width)
    #     start_i = int(median_index - num_indices)
    #     end_i = int(median_index + num_indices)
    #
    #     sorted_class = table[:, -1]  # pull class values
    #     best_gain = 0
    #     for t in range(start_i, end_i):
    #         ind = np.searchsorted(table[:, col], t)
    #         left = sorted_class[:ind]
    #         right = sorted_class[ind:]
    #
    #         # TODO: Edge case... lists sizes of 1 and 2
    #         score = gini_gain(sorted_class, [left, right])
    #         if score > best_gain:
    #             best_gain = score
    #             median_index = ind
    #             median = t
    #     """
    #     # -------------------------------
    #
    #     return median_index, table, median

    # # Experiment: Find split on MEAN
    # def find_split(self, table, col):
    #     table = table[table[:, col].argsort()]  # Sort by column
    #     mean = table[:, col].mean()
    #     mean_index = np.searchsorted(table[:, col], mean)
    #     return mean_index, table, mean

    # Experiment: Find split brute forcing on thresholds
    def find_split(self, table, col):
        best_i = 0
        best_gain = 0
        best_split_val = 0
        table = table[table[:, col].argsort()]  # Sort by column
        sorted_class = table[:, -1]  # pull class values

        thresh = []
        table_unique_in_col = np.unique(table[:, col])
        for r in range(0, table_unique_in_col.shape[0] - 1):
            mean = (table_unique_in_col[r] + table_unique_in_col[r + 1]) / 2
            thresh.append(mean)

        if len(thresh) == 0:
            thresh = np.arange(1, len(table[:, col]-1))

        for t in thresh:
            ind = np.searchsorted(table[:, col], t)
            left = sorted_class[:ind]
            right = sorted_class[ind:]

            # TODO: Edge case... lists sizes of 1 and 2
            score = gini_gain(sorted_class, [left, right])
            if score > best_gain:
                best_gain = score
                best_i = ind
                best_split_val = t

        median = np.median(table[:, col])
        ind = np.searchsorted(table[:, col], median)

        return ind, table, median

    def tree_build(self, features, classes, depth):

        # because there seems to be a weak type error in gradescope tests
        classes = np.asarray(classes)
        flat_class = classes.flatten().tolist()
        class_cnts = Counter(flat_class)
        class_list = list(class_cnts)

        # Edge case: Classes list is empty
        if len(class_list) == 0:
            return DecisionNode(None, None, None, None)

        # Base case 2: If a specified depth limit is reached, return a leaf labeled with
        # the most frequent class.
        # print("Depth: ", depth)
        # print(depth, " ", self.depth_limit)
        #print(depth)
        if depth >= self.depth_limit:
            most_common = class_cnts.most_common()[0][0]
            return DecisionNode(None, None, None, most_common)

        if len(class_list) == sum(class_list):
            # print("all sames")
            return DecisionNode(None, None, None, class_list[0])

        # Base case 1: If all elements of a list are of the same class, return a leaf node
        # with the appropriate class label.  WIKI: All the samples in the list belong to
        # the same class. When this happens, it simply creates a leaf node for the decision
        # tree saying to choose that class.
        if len(class_list) == 1:
            #
            # # TODO: --------------------------------------------------------------------------------------------
            # if len(class_list) == 0:
            #     return DecisionNode(None, None, None, 0)  # TODO: WHAT DO WE DO WITH EMPTY LIST>>> WHY EMPTY????
            # # TODO: --------------------------------------------------------------------------------------------

            return DecisionNode(None, None, None, int(class_list[0]))

        # 2,3) Let alpha_best be the attribute with the highest normalized gini gain.
        # For each attribute alpha: evaluate the normalized gini gain by splitting
        # on attribute alpha.

        # Calculate gini gain:
        # Concatenate features and classes together so we can sort values
        stacked_class = np.vstack(classes)
        table = np.concatenate([features, stacked_class], axis=1)
        bottle = []
        splits = []
        split_vals = []

        # Calculate ATTRIBUTE with the best gini.  (Least impure)
        for i_attr in range(0, features.shape[1]):

            # *Important* returns: table sorted on chosen column
            split_index, table, split_val = self.find_split(table, i_attr)
            sorted_class = table[:, -1]  # Sorts by class column

            gini = gini_gain(sorted_class, [sorted_class[:split_index], sorted_class[split_index:]])
            bottle.append(gini)
            splits.append(split_index)
            split_vals.append(split_val)

        # The best gini score calculated from trying all attributes
        alpha_best = max(bottle)
        best_attr_index = bottle.index(alpha_best)  # index of attr w/ best gini
        best_split = splits[best_attr_index]  # best split index
        best_split_val = split_vals[best_attr_index]  # best split val

        table = table[table[:, best_attr_index].argsort()]  # sort on best
        features = table[:, 0:features.shape[1]]
        sorted_class = table[:, -1]

        # 5) Repeat on the sublists obtained by splitting on alpha_best, and add those nodes
        # as children of this node

        features_left = features[:best_split]
        classes_left = sorted_class[:best_split]
        features_right = features[best_split:]
        classes_right = sorted_class[best_split:]

        if (classes_right.size == 0 and classes_left.size > 0) or (
                classes_left.size == 0 and classes_right.size > 0):
            most_common = class_cnts.most_common()[0][0]
            return DecisionNode(None, None, None, most_common)

        node = DecisionNode(None, None, None)
        node.left = self.tree_build(features_left, classes_left, depth + 1)
        node.right = self.tree_build(features_right, classes_right, depth + 1)
        node.decision_function = lambda f: f[best_attr_index] < best_split_val

        return node

    def classify(self, features):
        # print("classify")
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for f in features:
            # print("Deciding feature: ", f)
            decision = self.root.decide(f)
            class_labels.append(decision)

        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # Each dataset is a tuple of two arrays, (features, classes) consistent with what gets passed to all
    # your classifiers. Make sure they are NumPy arrays.

    # ([],[])

    folds = []
    stacked_class = np.vstack(dataset[1])
    table = np.concatenate([dataset[0], stacked_class], axis=1)

    for i in range(k):
        # shuffle
        np.random.shuffle(table)
        features = table[:, 0:4]
        classes = np.hstack(table[:, 4:]).tolist()

        split = table.shape[0] // k
        training_set = (table[0:split, 0:4], table[0:split, 4:])
        test_set = (table[split:, 0:4], table[split:, 4:])
        folds.append((test_set, training_set))

    return folds


class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        # Create multiple decision trees
        # INPUT: Data Set of size N with M dimension
        # 1) Sample n times from Data (use random sampling of training data)
        # 2) Sample m times from Attributes (use random sampling of attributes from that data)
        # 3) Learn Tree on sampled data and attributes
        # REPEAT UNTIL k trees

        n = features.shape[0]  # n = length of features
        m = features.shape[1]  # m = length of attributes

        stacked_class = np.vstack(classes)
        table = np.concatenate([features, stacked_class], axis=1)

        sample_size_n = int(n * self.example_subsample_rate)
        sample_size_m = int(m * self.example_subsample_rate)

        # when given unknown data... those trees vote on that result
        for j in range(self.num_trees):
            # Shuffle features by row, select random indices (w/ replacement) and extract
            np.random.shuffle(table)
            n_indices = np.random.randint(n, size=sample_size_n)  # features
            m_indices = np.random.choice(m, sample_size_m, replace=False)  # attrs
            samples = table[n_indices, :]
            train_classes = samples[:, 4]
            train_features = samples[:, m_indices]

            # Create new true with samples
            tree = DecisionTree(depth_limit=self.depth_limit)
            tree.fit(train_features, train_classes)
            self.trees.append(tree)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = []
        for f in features:
            # print("Deciding feature: ", f)
            voting = []
            for t in self.trees:
                voting.append(t.root.decide(f))
            decision = Counter(voting).most_common()[0][0]
            class_labels.append(decision)

        return class_labels


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        tmp_data = data
        data = data * data
        data = data + tmp_data
        return data

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        data = data[:100, :]
        sums = np.sum(data[:100, :], axis=1)
        imax = np.argmax(sums)
        return sums[imax], imax

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        flat = np.ravel(data)
        flat = flat[flat > 0]
        count_tuples = Counter(flat.tolist()).items()
        return count_tuples


class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self):
        self.root = None
        self.depth_limit = 10

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        return self.tree_build(features, classes, depth)

    # # Experiment: Find split on MEDIAN
    # def find_split(self, table, col):
    #     table = table[table[:, col].argsort()]  # Sort by column
    #     median_index = len(table[:, col]) // 2
    #     median = table[median_index, col]
    #     return median_index, table, median

    # Experiment: Find split brute forcing on thresholds
    def find_split(self, table, col):
        best_i = 0
        best_gain = 0
        best_split_val = 0
        table = table[table[:, col].argsort()]  # Sort by column
        sorted_class = table[:, -1]  # pull class values

        thresh = []
        table_unique_in_col = np.unique(table[:, col])
        for r in range(0, table_unique_in_col.shape[0] - 1):
            mean = (table_unique_in_col[r] + table_unique_in_col[r + 1]) / 2
            thresh.append(mean)

        if len(thresh) == 0:
            thresh = np.arange(1, len(table[:, col] - 1))

        for t in thresh:
            ind = np.searchsorted(table[:, col], t)
            left = sorted_class[:ind]
            right = sorted_class[ind:]

            # TODO: Edge case... lists sizes of 1 and 2
            score = gini_gain(sorted_class, [left, right])
            if score > best_gain:
                best_gain = score
                best_i = ind
                best_split_val = t

        median = np.median(table[:, col])
        ind = np.searchsorted(table[:, col], median)

        return ind, table, median

    def tree_build(self, features, classes, depth):

        # because there seems to be a weak type error in gradescope tests
        classes = np.asarray(classes)
        flat_class = classes.flatten().tolist()
        class_cnts = Counter(flat_class)
        class_list = list(class_cnts)

        # Edge case: Classes list is empty
        if len(class_list) == 0:
            return DecisionNode(None, None, None, None)

        if depth >= self.depth_limit:
            most_common = class_cnts.most_common()[0][0]
            return DecisionNode(None, None, None, most_common)

        if len(class_list) == sum(class_list):
            # print("all sames")
            return DecisionNode(None, None, None, class_list[0])

        if len(class_list) == 1:
            return DecisionNode(None, None, None, int(class_list[0]))

        stacked_class = np.vstack(classes)
        table = np.concatenate([features, stacked_class], axis=1)
        bottle = []
        splits = []
        split_vals = []

        # Calculate ATTRIBUTE with the best gini.  (Least impure)
        for i_attr in range(0, features.shape[1]):

            # *Important* returns: table sorted on chosen column
            split_index, table, split_val = self.find_split(table, i_attr)
            sorted_class = table[:, -1]  # Sorts by class column

            gini = gini_gain(sorted_class, [sorted_class[:split_index], sorted_class[split_index:]])
            bottle.append(gini)
            splits.append(split_index)
            split_vals.append(split_val)

        # The best gini score calculated from trying all attributes
        alpha_best = max(bottle)
        best_attr_index = bottle.index(alpha_best)  # index of attr w/ best gini
        best_split = splits[best_attr_index]  # best split index
        best_split_val = split_vals[best_attr_index]  # best split val

        table = table[table[:, best_attr_index].argsort()]  # sort on best
        features = table[:, 0:features.shape[1]]
        sorted_class = table[:, -1]

        # 5) Repeat on the sublists obtained by splitting on alpha_best, and add those nodes
        # as children of this node

        features_left = features[:best_split]
        classes_left = sorted_class[:best_split]
        features_right = features[best_split:]
        classes_right = sorted_class[best_split:]

        node = DecisionNode(None, None, None)
        node.left = self.tree_build(features_left, classes_left, depth + 1)
        node.right = self.tree_build(features_right, classes_right, depth + 1)
        node.decision_function = lambda f: f[best_attr_index] < best_split_val

        return node

    def classify(self, features):
        class_labels = []
        for f in features:
            # print("Deciding feature: ", f)
            decision = self.root.decide(f)
            class_labels.append(decision)

        return class_labels

def return_your_name():
    # return your name
    # TODO: finish this
    return 'dward45'
