import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.trees = []
        self.alphas = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        w = np.array([1/X.shape[0]] * X.shape[0])
        for i in range(0,self.n_weakers_limit):
            tree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                                min_samples_split=2,min_samples_leaf =1,
                                                min_weight_fraction_leaf=0.0, max_features=None,
                                                random_state=None, max_leaf_nodes=None,class_weight=None, presort=False)
            tree.fit(X,y,sample_weight = w)
            _y = tree.predict(X)
            error = 0
            count = 0
            for j in range(0, _y.shape[0]):
                if y[j] != _y[j]:
                    count += 1
                    error += w[j]
            print('error in the ' + str(i + 1) + 'th round is:', error)
            tmp = 1 - error
            alpha = 0.5 * np.log(tmp / error)
            z = 0
            for j in range(0,X.shape[0]):
                z += w[j] * np.exp(-alpha * y[j] * _y[j])
            for j in range(0,X.shape[0]):
                w[j] = (w[j] / z) * np.exp(-alpha * y[j] * _y[j])
            self.trees.append(tree)
            self.alphas.append(-alpha)
        return


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        for i in range(0, len(self.alphas)):
            if i == 0:
                y = self.alphas[i] * self.trees[i].predict(X)
            else:
                y += self.alphas[i] * self.trees[i].predict(X)
        return y


    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y = self.predict_scores(X)
        for i in range(0,len(y)):
            if y[i] >= threshold:
                y[i] = +1
            else:
                y[i] = -1
        return y

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
