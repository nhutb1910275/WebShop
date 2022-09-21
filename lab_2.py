import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

dt = pd.read_csv("winequality-red.csv", delimiter=';')

X = dt.iloc[:, 0:11]
y = dt.quality

# cau b
print(len(dt))
print(np.unique(dt.quality))
print(dt.quality.value_counts())
# cau c
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2/5.0, random_state=5)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# cau d
mohinh_KNN = KNeighborsClassifier(n_neighbors=7)
mohinh_KNN.fit(X_train, y_train)
y_pred = mohinh_KNN.predict(X_test)
# cau di
print("do chinh xac: ", accuracy_score(y_test, y_pred))

# cau dii
d_pred = mohinh_KNN.predict(X_test.iloc[0:8, :])
print("do chinh xac: ", accuracy_score(y_test.iloc[0:8], d_pred))

# cau e
model = GaussianNB()
model.fit(X_train, y_train)
thucte = y_test
dubao = model.predict(X_test)
model.predict_proba(X_test)
# print(thucte)
# print(dubao)
print("\ndo chinh xac: ", accuracy_score(thucte, dubao))
print(confusion_matrix(thucte, dubao))

# cau f
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3.0, random_state=10)
mohinh_KNN = KNeighborsClassifier(n_neighbors=5)
mohinh_KNN.fit(X_train, y_train)
y_pred = mohinh_KNN.predict(X_test)
print("do chinh xac: ", accuracy_score(y_test, y_pred))


model = GaussianNB()
model.fit(X_train, y_train)
thucte = y_test
dubao = model.predict(X_test)
model.predict_proba(X_test)
print("do chinh xac: ", accuracy_score(thucte, dubao))
