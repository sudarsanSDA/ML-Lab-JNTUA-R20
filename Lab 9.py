import numpy as np
import matplotlib.pyplot as plt

# Locally Weighted Linear Regression function
def locally_weighted_regression(X, y, tau=1.0):
    m, n = X.shape
    Y_hat = np.zeros(m)

    # For each data point
    for i in range(m):
        # Calculate the weight matrix based on tau
        weights = np.exp(-np.sum((X - X[i])**2, axis=1) / (2 * tau**2))
        
        # Apply weights as diagonal to the matrix
        W = np.diag(weights)
        
        # Compute theta using weighted least squares: theta = (X.T * W * X)^(-1) * X.T * W * y
        X_transpose_W = np.dot(X.T, W)
        theta = np.linalg.inv(np.dot(X_transpose_W, X)).dot(np.dot(X_transpose_W, y))
        
        # Predict for the current point
        Y_hat[i] = np.dot(X[i], theta)

    return Y_hat

# Generating some noisy data
np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)  # Feature
y = 2 * X.flatten() + 5 + np.random.normal(0, 2, X.shape[0])  # Linear relationship + noise

# Apply Locally Weighted Regression
y_hat = locally_weighted_regression(X, y, tau=1.0)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_hat, color='red', label='LWLR Fit')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Locally Weighted Linear Regression (LWLR)')
plt.legend()
plt.show()
