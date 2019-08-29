import numpy as np
import pandas as pd
import copy
import sys

# Perceptron Algorithm to find a good weight vector w, where t is the number of passes
def percept(p):
    print("Performing perceptron algorithm... ")
    # Read training data, get labels from last column and increase dimension by one
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
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
    labels = points.iloc[:, -1] # Get labels as a series
    points = points.drop(points.columns[-1], axis = 1) # Drop labels column from points
    mistakes = 0
    total_points = 0
    for index, row in points.iterrows():
        if labels[index] != predict(w, row):
            mistakes += 1
        total_points += 1
    return float(mistakes) / total_points

# Get the 5 most significant coordinates, if for_positive is True, we get the 5 in positive class, otherwise negative
def get_significant_coordinates(weights, for_positive):
    w = copy.deepcopy(weights)
    max_list = [] # Tuples of max values and its index
    for _ in range(5):  
        index = len(w)
        if for_positive:
            max = -sys.maxsize
        else:
            max = sys.maxsize
        for j in range(len(w)):
            if for_positive:     
                if w[j] > max: 
                    max = w[j]
                    index = j
            else:
                if w[j] < max:
                    max = w[j]
                    index = j
        w = np.delete(w, index) 
        max_list.append((max, index))
    return max_list

# Find error
if __name__ == '__main__':
    w2 = percept(2)
    w3 = percept(3)
    w4 = percept(4)
    w5 = percept(5)
    print(get_significant_coordinates(w5, False))
    print("2 pass on training: " + str(get_error(w2, 'pa3train.txt')))
    print("3 pass on training: " + str(get_error(w3, 'pa3train.txt')))
    print("4 pass on training: " + str(get_error(w4, 'pa3train.txt')))
    print("5 pass on training: " + str(get_error(w5, 'pa3train.txt')))
    print("2 pass on testing: " + str(get_error(w2, 'pa3test.txt')))
    print("3 pass on testing: " + str(get_error(w2, 'pa3test.txt')))
    print("4 pass on testing: " + str(get_error(w2, 'pa3test.txt')))
    print("5 pass on testing: " + str(get_error(w2, 'pa3test.txt')))
