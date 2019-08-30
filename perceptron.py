import numpy as np
import pandas as pd
import copy
import sys

# Perceptron Algorithm to find a good linear weight vector w, where p is the number of passes
def percept(p):
    print("Performing perceptron algorithm... ")
    # Read training data, get labels from last column
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
    # Initialize weight to a zero vector
    w = np.zeros(len(train.columns), dtype = int)
    # Do p passes of building the perceptron
    for _ in range(p):
        for index, row in train.iterrows():
            y = _y(labels, index)
            # Perform dot product. If it is <= 0, w_t+1 = w_t + (y_t * x_t)
            x = np.array(row)
            product = y * np.dot(w, x)
            if product <= 0:
                w = np.add(w, y * x)
    return w

# Perceptron Algorithm where k is the kernel function
def percept_kernelized(p, k):
    print("Performing perceptron kernel algorithm... ")
    # Read training data, get labels from last column
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
    # Initialize m as empty
    m = []
    # Do p passes of building the perceptron
    for _ in range(p):
        for index, row in train.iterrows():
            yt = _y(labels, index)
            # Do the kernelization
            kernel = 0
            for i in m:
                yi = _y(labels, i)
                kernel += yi * k(np.array(train.iloc[i]), np.array(row))
            if yt * kernel <= 0:
                m.append(index)
    return np.array(m)

# Get the label at a row. Arbitrarily, we will choose 1 as 1, and 2 as -1
def _y(labels, index):
    if labels[index] == 1:
        return 1
    else:
        return -1

# Exponential kernel function k(x,z) = e^(-||x-z||/20)
def _exponential(x, z):
    return np.e ** (-(np.linalg.norm(x - z) / 20))

# Polynomial kernel function k(x,z) = (<x,z> + 10)^2
def _polynomial(x, z):
    return (np.dot(x, z) + 10) ** 2

# Predict a label given the weight perceptron, where point is the row with features
def predict(w, point):
    x = np.array(point)
    pred = np.sign(np.dot(w, x))
    return _lab(pred)

# Predict a label given a kernelized perception, where point is the row with features
def predict_kernelized(m, point, k, labels):
    train = pd.read_csv('pa3train.txt', sep = r'\s+', header = None)
    labels = train.iloc[:, -1] # Get labels as a series
    train = train.drop(train.columns[-1], axis = 1) # Drop labels from training data
    sum = 0
    for i in m:
        yi = _y(labels, i)
        sum += yi * k(np.array(train.iloc[i]), np.array(point))
    pred = np.sign(sum)
    return _lab(pred)

# Return the label given the sign
def _lab(pred):
    if pred > 0:
        return 1
    elif pred == 0:
        return np.random.choice([1, 2])
    else: # Negative
        return 2

# Find training, validation, or test error
def get_error(f, filename, is_kernel, k): # k is only used for kernelized perceptrons
    points = pd.read_csv(filename, sep = r'\s+', header = None)
    labels = points.iloc[:, -1] # Get labels as a series
    points = points.drop(points.columns[-1], axis = 1) # Drop labels column from points
    mistakes = 0
    total_points = 0
    for index, row in points.iterrows():
        if (is_kernel):
            pred = predict_kernelized(f, row, k, labels)
        else:
            pred = predict(f, row)
        if labels[index] != pred:
            mistakes += 1
        total_points += 1
    return float(mistakes) / total_points

# Get the 5 most significant coordinates, if for_positive is True: return best 5 in positive class, otherwise negative
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
    we2 = percept_kernelized(2, _exponential)
    we3 = percept_kernelized(3, _exponential)
    we4 = percept_kernelized(4, _exponential)
    we5 = percept_kernelized(5, _exponential)
    wp2 = percept_kernelized(2, _polynomial)
    wp3 = percept_kernelized(3, _polynomial)
    wp4 = percept_kernelized(4, _polynomial)
    wp5 = percept_kernelized(5, _polynomial)
    print("2 pass on training: " + str(get_error(w2, 'pa3train.txt', False, 0)))
    print("3 pass on training: " + str(get_error(w3, 'pa3train.txt', False, 0)))
    print("4 pass on training: " + str(get_error(w4, 'pa3train.txt', False, 0)))
    print("5 pass on training: " + str(get_error(w5, 'pa3train.txt', False, 0)))
    print("2 pass on testing: " + str(get_error(w2, 'pa3test.txt', False, 0)))
    print("3 pass on testing: " + str(get_error(w3, 'pa3test.txt', False, 0)))
    print("4 pass on testing: " + str(get_error(w4, 'pa3test.txt', False, 0)))
    print("5 pass on testing: " + str(get_error(w5, 'pa3test.txt', False, 0)))
    print(get_significant_coordinates(w5, False))
    print("2 pass on training ex: " + str(get_error(we2, 'pa3train.txt', True, _exponential)))
    print("3 pass on training ex: " + str(get_error(we3, 'pa3train.txt', True, _exponential)))
    print("4 pass on training ex: " + str(get_error(we4, 'pa3train.txt', True, _exponential)))
    print("5 pass on training ex: " + str(get_error(we5, 'pa3train.txt', True, _exponential)))
    print("2 pass on testing ex: " + str(get_error(we2, 'pa3test.txt', True, _exponential)))
    print("3 pass on testing ex: " + str(get_error(we3, 'pa3test.txt', True, _exponential)))
    print("4 pass on testing ex: " + str(get_error(we4, 'pa3test.txt', True, _exponential)))
    print("5 pass on testing ex: " + str(get_error(we5, 'pa3test.txt', True, _exponential)))
    print("2 pass on training po: " + str(get_error(wp2, 'pa3train.txt', True, _polynomial)))
    print("3 pass on training po: " + str(get_error(wp3, 'pa3train.txt', True, _polynomial)))
    print("4 pass on training po: " + str(get_error(wp4, 'pa3train.txt', True, _polynomial)))
    print("5 pass on training po: " + str(get_error(wp5, 'pa3train.txt', True, _polynomial)))
    print("2 pass on testing po: " + str(get_error(wp2, 'pa3test.txt', True, _polynomial)))
    print("3 pass on testing po: " + str(get_error(wp3, 'pa3test.txt', True, _polynomial)))
    print("4 pass on testing po: " + str(get_error(wp4, 'pa3test.txt', True, _polynomial)))
    print("5 pass on testing po: " + str(get_error(wp5, 'pa3test.txt', True, _polynomial)))
