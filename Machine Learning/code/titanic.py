import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic = sns.load_dataset('titanic')
titanic = titanic.dropna(subset=['survived'])

# View the first few rows
print(titanic.head())
print(titanic.keys())
Y=np.array(titanic['survived'])
titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic['fare'] = titanic['fare'].fillna(titanic['fare'].median())
X=np.array([titanic['pclass'],
   titanic['age'],
   titanic['sibsp'],
   titanic['parch'],
   titanic['fare']]).T
X_train=X[:800]
X_test=X[800:]
Y_train=Y[:800]
Y_test=Y[800:]
model=LinearRegression()
model.fit(X_train,Y_train)
prediction=model.predict(X_test)
print(mean_squared_error(Y_test,prediction))
plt.scatter(range(len(Y_test)), Y_test, color='green', label='actual survived or not')
plt.scatter(range(len(prediction)), prediction, color='red', label='Predicted survive or not', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('survived')
plt.legend()
plt.title('titanic data')
plt.show()




