##############################################
# Programmer: Elizabeth Larson (starter code by Dr. Gina Sprint)
# Class: CPSC 322-01, Spring 2021
# Programming Assignment #6
# 04/14/21
# I didn't attempt the bonus. TODO: Edit (or not)
#
# Sources:
#   Shuffling a list (for train_test_split()): https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
#
# Description: This program computes functions that split a
#              dataset into trainig and test sets, performs
#              kfold cross validation (regular and stratified),
#              and generates a confusion matrix.
##############################################


import mysklearn.myutils as myutils
import numpy as np # For train_test_split() random numbers + all other "random" stuff

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    if random_state is not None:
       # Seed the random number generator
       np.random.seed(random_state)
    
    if shuffle:
        # Shuffle the rows in X and y before splitting maintaining the parallel order of X and y
        X_shuffled = []
        y_shuffled = []

        # Make a list of the indices and then shuffle them
        index_shuffled = list(range(len(X)))
        np.random.shuffle(index_shuffled)

        # Keep track of the new order of these values
        for i in index_shuffled:
            X_shuffled.append(X[i])
            y_shuffled.append(y[i])

        # Save the newly shuffled list in the original X and y (for further use)
        X = X_shuffled
        y = y_shuffled

    num_instances = len(X)
    if isinstance(test_size, float): # Find out the type
        test_size_not_ceiled = num_instances * test_size
        test_size = int((test_size_not_ceiled // 1) + 1) # Get ceiling
    split_index = num_instances - test_size

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    return X_train, X_test, y_train, y_test

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    X_train_folds = []
    X_test_folds = []

    # Populate the folds table with n_splits number of folds
    folds = []
    first_folds_stopping_point = len(X) % n_splits
    first_folds_size = (len(X) // n_splits) + 1
    other_folds_size = len(X) // n_splits

    curr_index = 0 # Keep track of the index we're looking at in X
    folded_data = 0 # Keep track of how many indecies we've taken note of

    # Load the first folds
    for _ in range(first_folds_stopping_point):
        new_fold = []
        for i in range(first_folds_size):
            new_fold.append(curr_index)
            curr_index += 1
            folded_data += 1
        folds.append(new_fold)
        
    # Load the rest of the folds
    while folded_data < len(X):
        new_fold = []
        for i in range(other_folds_size):
            new_fold.append(curr_index)
            curr_index += 1
            folded_data += 1
        folds.append(new_fold)
        
    # Keep track of the test fold and the training fold(s)
    for test_index in range(len(folds)):
        # Take note of the test folds
        X_test_folds.append(folds[test_index])
        
        # Take note of the training folds (which are everything BUT the test values)
        # Save it all in one entry (e.g. if you have a test set of [[3, 3], [4, 4]], save it as [3, 3, 4, 4])
        new_train_fold = [] # Keep track of all of the train values for this particular test case
        i = 0
        while i < n_splits:
            if i != test_index: # Only look at non test folds
                for j in range(len(folds[i])):
                    new_train_fold.append(folds[i][j])
            i += 1
        X_train_folds.append(new_train_fold)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """

    X_train_folds = []
    X_test_folds = []

    # Append n empty lists (folds) to a main list of folds
    folds = []
    for _ in range(n_splits):
        folds.append([])
        
    # Determine which labels we're working with by removing duplicates (e.g. [0, 0, 1, 1] -> [0, 1])
    y_no_repeats = []
    for y_value in y:
        if y_value not in y_no_repeats:
            y_no_repeats.append(y_value)

    # Take note of what the label for each data point
    X_sorted_by_y_value = []
    for _ in range(len(y_no_repeats)):
        X_sorted_by_y_value.append([]) # Fill X_sorted_by_y_value with n_splits empty lists
    for X_index in range(len(X)):
        for y_value in y_no_repeats:
            if y[X_index] == y_value: # The X_index we're looking at has a label that matches the y_value we're looking at
                X_sorted_by_y_value[y_no_repeats.index(y_value)].append(X_index)
                break # We've found the correct y value so stop searching for it
            
    first_folds_stopping_point = len(X) % n_splits
    first_folds_size = (len(X) // n_splits) + 1
    other_folds_size = len(X) // n_splits

    folded_data = 0 # Keep track of how many indecies we've taken note of

    # Now, store these values in their respective folds (e.g. if y_no_repeats is [0, 1], fold0 is for the 0-labeled values and fold1 is for the 1-labeled values)
    # Load the first folds
    folds_index = 0
    col = 0
    row = 0
    while folds_index < first_folds_stopping_point:
        if first_folds_size == len(folds[folds_index]):
            folds_index += 1
        while row < len(X_sorted_by_y_value):
            folds[folds_index].append(X_sorted_by_y_value[row][col])
            folded_data += 1
            row += 1
            if first_folds_size == len(folds[folds_index]):
                folds_index += 1
        row = 0
        col += 1

    # Load the rest of the folds
    while folded_data < len(X):
        if other_folds_size == len(folds[folds_index]):
            folds_index += 1
        while row < len(X_sorted_by_y_value):
            try:
                folds[folds_index].append(X_sorted_by_y_value[row][col])
                folded_data += 1
                row += 1
                if other_folds_size == len(folds[folds_index]):
                    folds_index += 1
            except IndexError:
                row += 1
        row = 0
        col += 1

    # Keep track of the test fold and the training fold(s)
    for test_index in range(len(folds)):
        # Take note of the test folds
        X_test_folds.append(folds[test_index])
            
        # Take note of the training folds (which are everything BUT the test values)
        # Save it all in one entry (e.g. if you have a test set of [[3, 3], [4, 4]], save it as [3, 3, 4, 4])
        new_train_fold = [] # Keep track of all of the train values for this particular test case
        i = 0
        while i < n_splits:
            if i != test_index: # Only look at non test folds
                for j in range(len(folds[i])):
                    new_train_fold.append(folds[i][j])
            i += 1
        X_train_folds.append(new_train_fold)

    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """

    matrix = []

    # Load the matrix with N rows (where N is the length of the labels and rows are lists of 0's)
    # e.g. 3 labels would build this matrix :[[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    N = len(labels)
    for _ in range(N):
        new_matrix_entry = []
        for _ in range(N):
            new_matrix_entry.append(0)
        matrix.append(new_matrix_entry)
        
    # Iterate through the labels as columns and rows of the matrix
    row = 0
    col = 0
    for row_label in labels:
        for col_label in labels:
            for i in range(len(y_true)): # Now, check the data tables index by index
                if y_true[i] == row_label and y_pred[i] == col_label:
                    matrix[row][col] += 1
            col += 1 # Move on to the next col
        row += 1 # Move to the next row and start over on the cols
        col = 0

    return matrix