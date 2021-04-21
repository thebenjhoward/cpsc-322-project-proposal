##############################################
# Programmer: Elizabeth Larson (starter code by Dr. Gina Sprint)
# Class: CPSC 322-01, Spring 2021
# Programming Assignment #6
# 04/14/21
# I didn't attempt the bonus.
#
# Sources:
#   Getting the key for sorting (using lambda) without importing "operator": https://docs.python.org/3/howto/sorting.html
#   Help with finding the value with the highest frequency in a dataset: https://www.tutorialspoint.com/find-most-frequent-element-in-a-list-in-python
#   Converting list values to a string (printing decision rules): https://stackoverflow.com/questions/12309976/how-do-i-convert-a-list-into-a-string-with-spaces-in-python/12309982
#   Appending to the beginning of a list: https://www.programiz.com/python-programming/methods/list/insert
# 
# Description: This program computes the logic for a linear regresser
#              class and a k neighbors class that mirrors Sci-kit Learn's
#              classes. They include fit, predicting, and finding nearest
#              k neighbors capabilities (as well as initializing data).
#              It also has fit and predict functions for Naive Bayes
#              Decision Tree classifiers (with a decision rules printer
#              for the latter).
##############################################


import mysklearn.myutils as myutils
import random # Imported for random number generating (see MyRandomClassifier predict()), because I wasn't sure how to do it without an import

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """

        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """

        # Calculate the x mean
        x_sum = 0
        for value in X_train:
            x_sum += value[0]
        mean_x = x_sum / len(X_train)
        
        # Calculate the y mean
        y_sum = 0
        for value in y_train:
            y_sum += value
        mean_y = y_sum / len(y_train)

        # Calculate slope and intercept
        # Convert 2D list (X_train) to 1D
        X_train_1D = []
        for value in X_train:
            X_train_1D.append(value[0])
        self.slope = sum([(X_train_1D[i] - mean_x) * (y_train[i] - mean_y) for i in range(len(X_train_1D))]) / sum([(X_train_1D[i] - mean_x) ** 2 for i in range(len(X_train_1D))])
        self.intercept = mean_y - self.slope * mean_x # y = mx + b => y - mx

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """

        y_predicted = []

        for value in X_test:
            y = (self.slope * value[0]) + self.intercept # y = mx + b
            y_predicted.append(y)

        return y_predicted


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """

        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """

        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """

        distances = []
        neighbor_indices = []
        
        # Get the distance each instance in train is to X_test
        X_test_counter = 0
        for i, instance in enumerate(self.X_train):
            instance.append(self.y_train[i]) # Append the label
            instance.append(i) # Append the original row index
            try:
                dist = myutils.compute_euclidean_distance(instance[:2], X_test[X_test_counter])
            except TypeError: # Handling the comparison of two categorical attributes (aka 2 strings)
                dist = myutils.calculate_distance_categorical_atts(self.X_train, self.y_train, X_test[X_test_counter])
            instance.append(dist) # Append the distance
        
        # Sort train by distance
        X_train_sorted = sorted(self.X_train, key=lambda data: data[-1]) # Sort by the last item in the row (the distance, in this case)
        k = self.n_neighbors # n_neighbors nearest neighbors
        top_k = X_train_sorted[:k]

        # Keep track of the distances and indicies of the nearest neighbors
        for row in top_k:
            distances.append(row[-1]) # -1st spot is the euc. distance
            neighbor_indices.append(row[-2]) # -2nd spot is the index
            
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_predicted = []
        
        # Find the nearest k (3, in this case) neighbors
        _, neighbor_indices = self.kneighbors(X_test)

        # Use the indicies of these neighbors to find the class names
        class_names = []
        for i in range(len(self.y_train)):
            for j in neighbor_indices:
                if i == j: # Index we're checking in y_train is the same as the index we're checking in neighbor_indices
                    class_names.append(self.y_train[i])
        
        # Take note of the value that occurs most frequently
        for i in range(len(X_test)):
            y = max(set(class_names), key = class_names.count)
            y_predicted.append(y)
            
        return y_predicted

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """

        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        
        # Just for future reference
        self.X_train = X_train
        self.y_train = y_train

        # For organization, create a seperate list of possible values (i.e. iPad trace should be ["no", "yes"])
        att_values = []
        for y_train_index in range(len(y_train)):
            if not y_train[y_train_index] in att_values:
                att_values.append(y_train[y_train_index])

        # Find the col number we're working on (e.g. buys_iphone is col 2)
        att_col_num = 0
        for col in range(len(X_train[0])):
            for att_value in att_values:
                if X_train[0][col] == att_value:
                    att_col_num = col
                    break
            # Not found at all, this isn't the index we're looking for, go to the next col

        # Calculate the priors (how often each value shows up in the dataset PREDICTING attribute)
        den_per_att = [] # Keep track of the number of occurances of each predicting att (parallel to att_values, i.e. yes = 5 and no = 3)... used later
        self.priors = []
        den = len(X_train) # Denominator
        for att_value in att_values:
            num = 0 # Numerator
            for X_train_index in range(len(X_train)):
                if X_train[X_train_index][att_col_num] == att_value:
                    num += 1
            den_per_att.append(num)
            self.priors.append(num/den)

        # For organization, create a seperate list of possible values for all other attributes
        other_att_values = []
        for col_index in range(len(X_train[0])):
            if col_index != att_col_num: # Skip the y_train col
                already_found_in_curr_att = [] # So we have no repeats, and resets to [] for each att
                new_att_values = [] # Stores all of the info for one att (e.g. att1 for iPad trace would be [4/5, 1/5])
                for X_train_index in range(len(X_train)):
                    if not X_train[X_train_index][col_index] in already_found_in_curr_att:
                        new_att_values.append(X_train[X_train_index][col_index])
                        already_found_in_curr_att.append(X_train[X_train_index][col_index])
                other_att_values.append(new_att_values)

        # Calculate the posteriors (how often each value shows up in the dataset for the OTHER attibutes)
        self.posteriors = []
        for att_count in range(len(other_att_values)): # [[1, 2], [5, 6]]
            for att_value_count in range(len(other_att_values[att_count])): # Inside att_count (aka [1, 2] and [5,6])
                new_post = []
                for class_value in att_values: # ["yes", "no"]
                    num = 0 # Numerator
                    den = den_per_att[att_values.index(class_value)] # Denominator
                    for row in X_train:
                        if row[att_col_num] == class_value and row[att_count] == other_att_values[att_count][att_value_count]: # result=yes and att1=1
                            num += 1
                    if den > 0:
                        new_post.append(num/den)
                    else:
                        new_post.append(0.0)
                self.posteriors.append(new_post)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        y_predicted = []

        # For organization, create a seperate list of possible values (i.e. iPad trace should be ["no", "yes"])
        # So we can store values like "yes" or "no" later
        att_values = []
        for y_train_index in range(len(self.y_train)):
            if not self.y_train[y_train_index] in att_values:
                att_values.append(self.y_train[y_train_index])

        # Find the col number we're working on (e.g. buys_iphone is col 2)
        att_col_num = 0
        for col in range(len(self.X_train[0])):
            for att_value in att_values:
                if self.X_train[0][col] == att_value:
                    att_col_num = col
                    break
            # Not found at all, this isn't the index we're looking for, go to the next col

        # For organization, create a seperate list of possible values for all other attributes
        other_att_values = []
        for col_index in range(len(self.X_train[0])):
            if col_index != att_col_num: # Skip the y_train col
                already_found_in_curr_att = [] # So we have no repeats, and resets to [] for each att
                new_att_values = [] # Stores all of the info for one att (e.g. att1 for iPad trace would be [4/5, 1/5])
                for X_train_index in range(len(self.X_train)):
                    if not self.X_train[X_train_index][col_index] in already_found_in_curr_att:
                        new_att_values.append(self.X_train[X_train_index][col_index])
                        already_found_in_curr_att.append(self.X_train[X_train_index][col_index])
                other_att_values.append(new_att_values)

        # Find the posterior row numbers that we'll be checking
        rows_counted = 0
        post_rows_to_check = []
        for X_test_index in range(len(X_test)):
            for att_index in range(len(other_att_values)):
                for value in other_att_values[att_index]:
                    if value == X_test[X_test_index][att_index]:
                        post_rows_to_check.append(rows_counted)
                    rows_counted += 1

        # Go through each case that the predicted class could be (result=yes and result=no)
        calculated_probs = []
        for prior in self.priors: # Grab the probability of that result happening in the whole dataset (from the priors)
            is_first_run_through = True
            no_class_prob = 0 # Equation without the P(class=value)
            # Grab the probability of that result happening in the X_test case (att1=1, att2=5, AND result=yes)
            # Assume conditional dependence: P(att1=1 & result=yes) * P(att2=5 & result=yes)
            for row_to_check in post_rows_to_check:
                new_prob_value = self.posteriors[row_to_check][self.priors.index(prior)]
                if is_first_run_through: # First time through? Don't multiple by 0
                    no_class_prob += new_prob_value
                    is_first_run_through = False
                else: # Any other time through? Go right ahead
                    no_class_prob = no_class_prob * new_prob_value
            # Calulate total probability
            total_prob = no_class_prob * prior # P(att1=1 & result=yes) * P(att2=5 & result=yes) * P(result=yes)
            calculated_probs.append(total_prob)

        # Now that we're done traversing through the possible class values, see which one is largest (that's our winner)
        pred_class_value_index = calculated_probs.index(max(calculated_probs)) # Keep track of its index too
        y_predicted.append(att_values[pred_class_value_index])

        return y_predicted

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """

        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """

        self.X_train = X_train
        self.y_train = y_train

        # Calculate headers (e.g. ["att0", "att1", ...])
        headers = []
        for col_index in range(len(X_train[0])):
            headers.append("att" + str(col_index))

        # Calculate the attribute domains dictionary (e.g. standing can be 1 or 2)
        domains = {}
        for col_index in range(len(X_train[0])):
            att_values = []
            for row in X_train:
                if not row[col_index] in att_values:
                    att_values.append(row[col_index])
            att_values.sort() # Put them in alphabetical order
            domains[headers[col_index]] = att_values
        
        # Stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = headers.copy()

        # Initial tdidt() call
        tree = myutils.tdidt(train, available_attributes, headers, domains)
        self.tree = tree
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        y_predicted = []
        for X_test_value in X_test:
            prediction = myutils.predict_recursive_helper(self.tree, X_test_value)
            if prediction == None: # Pick a random row and use that as the prediction
                random_index = random.randint(0, len(self.y_train))
                prediction = self.y_train[random_index]
            y_predicted.append(prediction)

        return y_predicted

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        rules = []

        if attribute_names == None:
            rule = ["IF", self.tree[1], "=="]
        else: # [-1]st spot of att labels is the index
            att_index = int(self.tree[1][-1])
            rule = ["IF", attribute_names[att_index], "=="]
        
        index = 2 # Values start at index 2
        
        while index < len(self.tree): # Go through the values on each partition (e.g. Junior, Mid and Senior)
            # Calculate the branch (initial attribute that's already in there is passed in as rule)
            rule = myutils.make_rule_for_one_branch(self.tree[index], attribute_names, class_name, rule)

            rules.append(rule)
            if type(rule[0]) == list: # Returned a list of lists (often found in inner tree attribuet splits)
                for rule_value in rule:
                    rules.append(rule_value)
                rules.remove(rule)
            index += 1
            if index < len(self.tree): # Check if we've hit the end of the tree (and if so, don't add any more rules)
                rule = []
                if attribute_names == None:
                    rule = ["IF", self.tree[1], "=="]
                else: # [-1]st spot of att labels is the index
                    att_index = int(self.tree[1][-1])
                    rule = ["IF", attribute_names[att_index], "=="]

        # Now print all of the rules
        for rule in rules:
            # But first make sure every value is a string
            for rule_value_index in range(len(rule)):
                if not type(rule[rule_value_index]) is str:
                    rule[rule_value_index] = str(rule[rule_value_index])

            # Rules are currently a list of lists, make them each one long string so they're easier to read
            rule = " ".join(rule)
            print(rule)

    # BONUS METHOD
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

        Notes: 
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """

        pass
