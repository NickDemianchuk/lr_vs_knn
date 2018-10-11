import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from data_cleaner import DataCleaner
from knn import KNN

df = pd.read_csv('train.csv')
dc = DataCleaner()
df = dc.clean_data(df)

X = df.values[:, 1:]
X = normalize(X, axis=0, norm='max')
y = df.values[:, 0]

# Splitting dataset into 70% train and 30% temporary dataset that needs to be splitted
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Splitting temporary dataset into 50% test and 50% development
X_dev, X_test, y_dev, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=0)

# lr = LogisticRegression()
lr = LogisticRegression(C=1000.0)
lr.fit(X_train, y_train)
lr_dev_accuracy = accuracy_score(y_dev, lr.predict(X_dev))
print('The accuracy of LR classifier evaluated on the development data is %.3f' % lr_dev_accuracy)

knn = KNN(X_train, y_train, k=10)
knn_dev_accuracy = accuracy_score(y_dev, knn.predict(X_dev))
print('The accuracy of KNN classifier evaluated on the development data is %.3f' % knn_dev_accuracy)

lr_test_accuracy = accuracy_score(y_test, lr.predict(X_test))
knn_test_accuracy = accuracy_score(y_test, knn.predict(X_test))
print('The accuracy of LR evaluated on the test data is %.3f' % lr_test_accuracy)
print('The accuracy of KNN evaluated on the test data is %.3f' % knn_test_accuracy)
print('KNN beats LR by %.3f percent' % (knn_test_accuracy - lr_test_accuracy))
