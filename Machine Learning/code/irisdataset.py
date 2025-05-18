import mglearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error

dataset=load_iris()
print(dataset)
X_train,X_test,Y_train,Y_test=train_test_split(dataset['data'],dataset['target'],random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
model=knn.fit(X_train,Y_train)
prediction=model.predict(X_test)
print(prediction)
print(mean_squared_error(Y_test,prediction))
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(f"predicted target:{prediction}\n predicted name is:{dataset.target_names[prediction]}")
