import operator

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from knn import KNN


class CLSTuner(object):

    def __init__(self):
        self.best_acc = 0
        self.best_cls = None

    def get_tuned_LR(self, X_train, y_train, X_test, y_test):
        self.best_acc = 0
        self.best_cls = None
        for i in range(-4, 5):
            for penalty in ['l1', 'l2']:
                for class_weight in [None, 'balanced']:
                    lr = LogisticRegression(C=10 ** i, penalty=penalty, class_weight=class_weight)
                    lr.fit(X_train, y_train)
                    acc = accuracy_score(y_test, lr.predict(X_test))
                    self.compare_classifiers(lr, acc)

        print('The accuracy of the tuned LR evaluated on the dev set is %.3f percent' % (self.best_acc * 100))
        print('The parameters of the tuned LR:\n' + str(self.best_cls))
        return self.best_cls

    def get_tuned_KNN(self, X_train, y_train, X_test, y_test):
        self.best_acc = 0
        self.best_cls = None
        # List for holding k value and gained accuracy
        knn_params = []
        for i in range(1, 21):
            knn = KNN(X_train, y_train, k=i)
            acc = accuracy_score(y_test, knn.predict(X_test))
            self.compare_classifiers(knn, acc)
            knn_params.append([i, acc])

        # Sorting the list in descending order by accuracy
        knn_params.sort(key=operator.itemgetter(1))

        # Converting into numpy array for convenience
        knn_params = np.array(knn_params)

        # Building a plot
        plt.scatter(knn_params[:, 0], knn_params[:, 1],
                    color='lightgreen', marker='o', label='k values')
        plt.scatter(knn_params[-1, 0], knn_params[-1, 1],
                    color='g', edgecolors='black', marker='o', label='The best k value')
        plt.title('')
        plt.xlabel('Number of neighbors')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        print('\nThe accuracy of the tuned KNN (k=' + str(int(knn_params[-1, 0])) +
              ') evaluated on the dev set is %.3f percent' % (self.best_acc * 100))
        return self.best_cls

    def compare_classifiers(self, cls, acc):
        if self.best_acc < acc:
            self.best_cls = cls
            self.best_acc = acc
        return self.best_cls
