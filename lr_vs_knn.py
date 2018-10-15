import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_cleaner import DataCleaner
from knn import KNN

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

# Splitting dataset into 70% train and 30% temporary dataset that needs to be splitted
X_train, X_temp, y_train, y_temp = train_test_split(
    X_std, y, test_size=0.3, random_state=0)

# Splitting temporary dataset into 50% test and 50% development
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0)

# Training and evaluating LR classifier
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_dev_accuracy = accuracy_score(y_dev, lr.predict(X_dev)) * 100
print('The accuracy of LR evaluated on the development data is %.3f percent' % lr_dev_accuracy)

# Evaluating KNN classifier
knn = KNN(X_train, y_train, k=15)
knn_dev_accuracy = accuracy_score(y_dev, knn.predict(X_dev)) * 100
print('The accuracy of KNN evaluated on the development data is %.3f percent' % knn_dev_accuracy)

lr_test_accuracy = accuracy_score(y_test, lr.predict(X_test)) * 100
knn_test_accuracy = accuracy_score(y_test, knn.predict(X_test)) * 100
print('\nThe accuracy of LR evaluated on the test data is %.3f percent' % lr_test_accuracy)
print('The accuracy of KNN evaluated on the test data is %.3f percent' % knn_test_accuracy)
print('KNN beats LR by %.3f percent' % (knn_test_accuracy - lr_test_accuracy))
