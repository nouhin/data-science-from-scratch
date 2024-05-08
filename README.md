# data-science-from-scratch
## K-Nearest Neighbors
- K-Nearest Neighbors is a non-parametric method. It is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure.
- The algorithm works as follows:
    1. Calculate the distance between the new data point and all data points in the training set.
    2. Sort the distances and determine the k-nearest neighbors based on the k-th minimum distance.
    3. Gather the categories of the k-nearest neighbors.
    4. Use the majority vote to assign the category to the new data point.
- Commonly used similarity measures include Euclidean distance and Cosine similarity.
- For regression problems, the output is the average of the values of k-nearest neighbors. For classification problems, the output is the mode of the classes of k-nearest neighbors. 
- The optimal value of k can be found using cross-validation.
    - Simplest approach: k = sqrt(N), where N is the number of data points.
    - Cross-validation: Split the data into training and validation sets. Try different values of k and choose the one that gives the best performance on the validation set.
- K-Nearest Neighbors is sensitive to irrelevant features and the scale of the data.
- Time complexity is O(MN) + O(N logN). N is the number of data points and M is the number of features.
- Space complexity is O(N).

## Logistic Regression
- Logistic Regression is a linear model for binary classification problems.Logistic function is a sigmoid function that maps any real value into the range [0,1].
- The algorithm works as follows:
    1. Initialize the weights and bias.
    2. Calculate the linear combination of weights and input features.
    3. Apply the sigmoid function to the result.
    4. Calculate the loss using the log loss function.
    5. Update the weights and bias using gradient descent.
    6. Repeat steps 2-5 until convergence.

## K-Means
- K-Means is a clustering algorithm that partitions the data into K clusters.
- The algorithm works as follows:
    1. Initialize K centroids randomly.
    2. Assign each data point to the nearest centroid.
    3. Update the centroids by taking the mean of all data points assigned to that centroid.
    4. Repeat steps 2 and 3 until convergence.


