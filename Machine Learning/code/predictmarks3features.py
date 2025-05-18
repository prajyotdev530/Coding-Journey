import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Extended dataset with 3 features: study hours, play hours, sleep hours
dataset = {
    'studyhours': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    'playhours':  [2.0, 2.5, 3.0, 3.5, 3.0, 4.0, 3.5, 4.0, 4.5, 5.0],
    'sleephours': [7.0, 6.5, 6.0, 5.5, 6.0, 5.0, 6.5, 6.0, 5.5, 5.0],
    'marks':      [1.5, 3.8, 6.1, 8.5, 10.9, 13.3, 15.6, 18.1, 20.4, 22.7]
}

# Convert to numpy array with multiple features (shape: 10 samples x 3 features)
X = np.array([
    dataset['studyhours'],
    dataset['playhours'],
    dataset['sleephours']
]).T  # transpose to get shape (10, 3)

y = np.array(dataset['marks'])

# Train-test split: first 7 samples for training, last 3 for testing
X_train = X[:7]
X_test = X[7:]
y_train = y[:7]
y_test = y[7:]

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print coefficients and intercept
print("Coefficients (slopes):", model.coef_)
print("Intercept:", model.intercept_)

# Mean squared error
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Plot actual vs predicted marks for test data
plt.scatter(range(len(y_test)), y_test, color='green', label='Actual Marks')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Marks', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Marks')
plt.legend()
plt.title('Actual vs Predicted Marks on Test Data')
plt.show()