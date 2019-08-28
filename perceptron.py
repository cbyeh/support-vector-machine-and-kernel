import numpy as np
import pandas as pd


# Perceptron Algorithm to find a good weight w, where t is the number of passes
def percept(t):
    print("Performing perceptron algorithm... ")
    # Read training data, get labels from last column and increase dimension by one
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
    train[len(train.columns)] = 1 # Extra dimension for hyperplane calculation
    # Initialize weight to a zero vector
    w = np.zeros(len(train.columns), dtype = int)
    # Do t passes of building the perceptron
    for _ in range(t):
        for index, row in train.iterrows():
            # Get the label at each row. Arbitrarily, we will choose 1 as 1, and 2 as -1
            if labels[index] == 1:
                y = 1
            else:
                y = -1
            x = np.array(row)
            product = y * np.dot(w, x)
            # print(product)
            print(len(w))
            print(len(x))

# Given a kernal function,


percept(1)











# Predict a label given the root of the tree, where point is the row with features
def predict(tree, point):
    curr = tree
    while not curr.is_leaf:
        index, split = curr.rules
        if (point[index] <= split):
            curr = curr.left_child
        else:
            curr = curr.right_child
    return curr.predict

# Find training, validation, or test error
def get_error(tree, filename):
    points = pd.read_csv(filename, sep = r'\s+', header = None)
    mistakes = 0
    total_points = 0
    for _, row in points.iterrows():
        if row[22] != predict(tree, row):
            mistakes += 1
        total_points += 1
    return float(mistakes) / total_points
