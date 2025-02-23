import numpy as np
import DTLearner as dt


class BagLearner(object):

    def __init__(self, learner=dt.DTLearner, kwargs={}, bags=20, boost=False, verbose=False):

        self.learners = []
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        for l in range(0, bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        return 'dward45'

    def addEvidence(self, x_train, y_train):

        for l in self.learners:
            rand_bag = [None] * x_train.shape[0]
            for i in range(0, len(rand_bag)):
                rand_bag[i] = (np.random.random() > 0.40)
            extract_x = x_train[rand_bag]
            extract_y = y_train[rand_bag]
            l.addEvidence(extract_x, extract_y)

    def query(self, x_test):
        arr = []
        for l in self.learners:
            arr.append(l.query(x_test))
        results = np.array(arr)
        result = np.mean(results, axis=0)
        return result


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
