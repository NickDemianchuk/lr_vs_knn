import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from cls_tuner import CLSTuner
from data_cleaner import DataCleaner

# Reading a csv file
df = pd.read_csv('train.csv')

# Preprocessing data
dc = DataCleaner()
df = dc.clean_data(df)

# Splitting data into dataset and labels
X = df.values[:, 1:]
y = df.values[:, 0]

# Standardizing the dataset
sc = StandardScaler()
sc.fit(X)
X_std = sc.transform(X)

# Splitting dataset into 70% train and 30% temporary dataset that needs to be split
X_train, X_temp, y_train, y_temp = train_test_split(
    X_std, y, test_size=0.3, random_state=0)

# Splitting temporary dataset into 50% test and 50% development
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0)

# Training and evaluating LR on the dev set
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_dev_accuracy = accuracy_score(y_dev, lr.predict(X_dev)) * 100
print('\nThe accuracy of the LR (default params) evaluated on the dev set is %.3f percent' % lr_dev_accuracy)

# Tuning and evaluating LR on the dev set
cls_tuner = CLSTuner()
lr = cls_tuner.get_tuned_LR(X_train, y_train, X_dev, y_dev)

# Tuning and evaluating KNN on the dev set
knn = cls_tuner.get_tuned_KNN(X_train, y_test, X_dev, y_dev)

# Evaluating tuned LR and KNN on the test set
lr_test_acc = accuracy_score(y_test, lr.predict(X_test)) * 100
knn_test_acc = accuracy_score(y_test, knn.predict(X_test)) * 100
print('\nThe accuracy of the tuned LR on the test set is %.3f percent' % lr_test_acc)
print('The accuracy of the tuned KNN on the test set is %.3f percent' % knn_test_acc)
print('LR performs better than KNN by %.3f percent' % (lr_test_acc - knn_test_acc))
