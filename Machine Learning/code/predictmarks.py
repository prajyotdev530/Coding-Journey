import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
dataset={'studyhours':[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]],
         'marks':[1.5, 3.8, 6.1, 8.5, 10.9, 13.3, 15.6, 18.1, 20.4, 22.7]}
studyhours_X_train=dataset['studyhours'][0:7]
studyhours_X_test=dataset['studyhours'][7:10]
marks_Y_train=dataset['marks'][0:7]
marks_Y_test=dataset['marks'][7:10]
model=linear_model.LinearRegression()
model.fit(studyhours_X_train,marks_Y_train)
prediction=model.predict(studyhours_X_test)
print(mean_squared_error(marks_Y_test,prediction))
print(prediction)
