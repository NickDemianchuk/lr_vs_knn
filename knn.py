import math
import operator

import numpy as np


class KNN(object):

    def __init__(self, X_train, y_train, k=5):
        self.X_train = X_train
        self.y_train = y_train
        self.k = k

    def predict(self, X_test):
        predict = []
        for sample in X_test:
            neighbors = self.find_k_neighbors(sample)
            predict.append(self.get_class_label(neighbors))
        return np.array(predict)

    # Finding k closest neighbors to specified sample
    def find_k_neighbors(self, sample):
        # Creating a list of lists to store actual class labels
        # and distances between test and train samples
        distances = []
        for i in range(self.y_train.size):
            distances.append((self.y_train[i], self.get_distance(self.X_train[i], sample)))

        # Sorting the list in descending order
        distances.sort(key=operator.itemgetter(1))

        # Creating a list of first k neighbors' class labels
        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        return np.array(neighbors)

    # Euclidean distance between samples x1 and x2
    def get_distance(self, x1, x2):
        dist = 0.0
        for i in range(x1.size):
            dist += pow((x1[i] - x2[i]), 2)
        return math.sqrt(dist)

    # In case of tie method will return a negative class label
    def get_class_label(self, neighbors):
        return 1 if neighbors.mean() > 0 else -1
