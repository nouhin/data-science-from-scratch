from math import sqrt


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


if __name__ == '__main__':
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [0, 1, 0, 1, 0]

    knn = KNN(k=2)
    knn.fit(X, y)
    pred = knn.predict(X_pnt=[0, 0])
    print(pred)  # Output: 0
