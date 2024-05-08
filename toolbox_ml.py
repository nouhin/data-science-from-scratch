from math import sqrt

import numpy as np


class KNN:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k

    def fit(self, X: list, y: list):
        """Store the data points and their labels"""
        self.X = X
        self.y = y

    def predict(self, X_pnt: list):
        """Predict the label/class for a given data point X_pnt using the k-nearest neighbor algorithm"""
        distances_labels = [(self._distance(X_pnt, point), label) for point, label in zip(self.X, self.y)]
        neighbors = sorted(distances_labels, key=lambda x: x[0])[:self.k]
        neighbors_labels = [label for _, label in neighbors]
        return self._get_mode(neighbors_labels)

    def _distance(self, X1, X2):
        return sqrt(sum((x1 - x2)**2 for x1, x2 in zip(X1, X2)))

    def _get_mode(self, data: list):
        freq = {}
        for item in data:
            if item in freq.keys():
                freq[item] += 1
            else:
                freq[item] = 1
        max_freq = max(freq.values())
        modes = [key for key, value in freq.items() if value == max_freq]
        return modes[0]

    def _get_average(self, data: list):
        return sum(data) / len(data)


class LogisticRegression:
    """Logistic regression model using gradient descent to optimize the weights and bias."""
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize weights and bias
        n_samples, n_features = len(X), len(X[0])
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            # Gradient descent
            self._gradient_descent(X, y, n_samples)

        return self.weights, self.bias

    def predict(self, X):
        """Predict the class for a given data point."""
        y_pred = self._predict(X)
        y_label = [1 if v > 0.5 else 0 for v in y_pred]
        return y_label

    def _predict(self, X):
        """Calculate the linear model and apply the sigmoid function."""
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def _sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def _gradient_descent(self, X, y, n_samples):
        # Evaluate the loss
        error_array = y - self._predict(X)  # y is flattened
        # Calculate the gradients
        dw = np.dot(X.T, error_array) / n_samples
        db = np.sum(error_array) / n_samples
        # Update the weights and bias
        self.weights += self.lr * dw
        self.bias += self.lr * db

    def _accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([0, 0, 1, 1, 1])
    X_test = np.array([[2, 3]])
    lr = LogisticRegression(lr=0.001, epochs=10000)
    w, b = lr.fit(X, y)
    print(lr.predict(X_test))
