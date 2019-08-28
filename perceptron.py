import numpy as np
import pandas as pd

# Perceptron Algorithm to find a good weight vector w, where t is the number of passes
def percept(p):
    print("Performing perceptron algorithm... ")
    # Read training data, get labels from last column and increase dimension by one
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
    train[len(train.columns)] = 1 # Extra dimension for hyperplane calculation
    # Initialize weight to a zero vector
    w = np.zeros(len(train.columns), dtype = int)
    # Do p passes of building the perceptron
    for _ in range(p):
        for index, row in train.iterrows():
            # Get the label at each row. Arbitrarily, we will choose 1 as 1, and 2 as -1
            if labels[index] == 1:
                y = 1
            else:
                y = -1
            # Perform dot product. If it is <= 0, w_t+1 = w_t + (y_t * x_t)
            x = np.array(row)
            product = y * np.dot(w, x)
            if product <= 0:
                w = np.add(w, y * x)
    return w

# Given a kernal function,

# Predict a label given the weight perceptron, where point is the row with features
def predict(w, point):
    x = np.array(point)
    pred = np.sign(np.dot(w, x))
    if pred > 0:
        return 1
    elif pred == 0:
        return np.random.choice([1, 2])
    else: # Negative
        return 2

# Find training, validation, or test error
def get_error(w, filename):
    points = pd.read_csv(filename, sep = r'\s+', header = None)
    mistakes = 0
    total_points = 0
    for _, row in points.iterrows():
        if row.iloc[-1] != predict(w, row):
            mistakes += 1
        total_points += 1
    return float(mistakes) / total_points

# Find error
if __name__ == '__main__':
    w = percept(1)
    print(get_error(w, 'pa3train.txt'))
