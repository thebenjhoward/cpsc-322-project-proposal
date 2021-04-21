##############################################
# Programmer: Elizabeth Larson (starter code by Dr. Gina Sprint)
# Class: CPSC 322-01, Spring 2021
# Programming Assignment #6
# 04/14/21
# I didn't attempt the bonus.
#
# Sources:
#       Checking if all list values are the same (case 2 decision trees): https://www.geeksforgeeks.org/python-check-if-all-elements-in-a-list-are-same/
# 
# Description: This program computes reusable general-purpose functions
##############################################


import random # For majority voting leaf node "flip a coin" solution (if the clash result is 50%/50%)

import math # For log calculations

def compute_euclidean_distance(v1, v2):
    """Calculate the euclidean distance between two vectors

    Args:
        v1(list of numeric vals): First vector
        v2(list of numeric vals): Second vector
        
    Returns:
        dist(float): The distance between the vectors
    """
        
    # Just look at the first two points in the vectors
    new_v1 = [v1[0], v1[1]]
    new_v2 = [v2[0], v2[1]]
        
    assert len(new_v1) == len(new_v2)
    dist = (sum([(new_v1[i] - new_v2[i]) ** 2 for i in range(len(new_v1))])) ** (1/2) # Get the square root by taking this formula to the 1/2 power
    return dist

def convert_to_DOE(values):
    """Convert a list of values (MPG, for this dataset) to the DOE values listed in the table in step 1's notebook/write-up
    
        Args:
            values (list of float): The values were are converting
            
        Returns:
            converted_values (list of int): These conversted values on a scale of [1-10]
    """
    converted_values = []
    
    for value in values:
        if value <= 13: # ≤13
            converted_values.append(1)
        elif value > 13 and value <= 14: # 14
            converted_values.append(2)
        elif value > 14 and value <= 16: # 15-16
            converted_values.append(3)
        elif value > 16 and value <= 19: # 17-19
            converted_values.append(4)
        elif value > 19 and value <= 23: # 20-23
            converted_values.append(5)
        elif value > 23 and value <= 26: # 24-26
            converted_values.append(6)
        elif value > 27 and value <= 30: # 27-30
            converted_values.append(7)
        elif value > 30 and value <= 36: # 31-36
            converted_values.append(8)
        elif value > 36 and value <= 44: # 37-44
            converted_values.append(9)
        elif value < 44: # ≥45
            converted_values.append(10)
            
    return converted_values

def normalize_data(values):
    """Normalize a group of values to a 0.0-1.0 scale

    Args:
        values(list of obj): Data we want to normalize
        
    Returns:
        noramlized_values(float): These values after calulations
    """
    
    normalized_values = []
    
    # value - smallest value in the dataset
    min_value = min(values)
    for index in range(len(values)):
        normalized_values.append(values[index] - min_value)

    # value / largest value in the dataset
    max_value = max(normalized_values)
    for index in range(len(normalized_values)):
        normalized_values[index] = normalized_values[index] / max_value
    
    return normalized_values

def calculate_accuracy_and_error_rate(matrix):
    """Uses a confusion matrix to determine the amount of correct and incorrect guesses
    Use these values to compute the accuracy and error rate of the matrix

    Args:
        matrix(list of list of obj): The confusion matrix we're checking
        
    Returns:
        accuracy(float): How many guesses were correct (decimal form of %)
        error_rate(float): How many guesses were incorrect (decimal form of %)
    """
    
    # Add up all values in the datasets
    total = 0.0
    for value in matrix:
        for value_index in range(len(value)):
            total += value[value_index]

    if total != 0.0: # Only do this calulating if there was at least one correct prediction
        # Keep track of the correctly guessed ones (where actual is 1 and pred is 1 and so on)
        # Also keep track of incorrect guesses: times when the predicted guessed
        correct_guesses_total = 0
        incorrect_guesses_total = 0
        for row_index in range(len(matrix)):
            for col_index in range(len(matrix[row_index])):
                if (row_index + 1) == (col_index + 1): # e.g. row_index=0 and col_index=0 would be the pairing for predicting 1 and being right... the diagonal from 0,0 to N,N on the matrix
                    correct_guesses_total += matrix[row_index][col_index]
                    break # Now stop checking the cols and go to the next row
                elif matrix[row_index][col_index] != 0: # Skip 0 values because these aren't predictions
                    incorrect_guesses_total += matrix[row_index][col_index]

        # Now, calculate the accuracy and error rate
        accuracy = correct_guesses_total / total
        error_rate = incorrect_guesses_total / total
    else: # Nothing was correct
        accuracy = 0.0
        error_rate = 1.0
    
    return accuracy, error_rate

def calculate_distance_categorical_atts(X, y, v):
    """Calculate the predicting class of a vector that's categorical

    Args:
        X(list of list of obj): X_train (the dataset)
        y(list of obj): y_train (col we're predicting on)
        v(list of numeric vals): Vector values
        
    Returns:
        dist(obj): The predicted value
    """
    
    # Go through each row in X and find the "closest" value (i.e. the attribute with the most matching values)
    num_matching_atts = []
    for row_index in range(len(X)):
        matching_atts_count = 0
        for col_index in range(len(v)):
            if v[col_index] == X[row_index][col_index]: # Found a match!
                matching_atts_count += 1
        num_matching_atts.append(matching_atts_count)
            
    # Find the row that has the most matches on it
    row_with_most_matching_atts = num_matching_atts.index(max(num_matching_atts))
    dist = y[row_with_most_matching_atts]
    
    return dist

def all_same_class(instances):
    """Check if all instance labels match the first label

    Args:
        instances(list of lists): Instance set we're checking
        
    Returns:
        True or False, depending on if all of the instance labels match the first label
    """

    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # Otherwise, all instance labels matched the first label

def select_attribute(instances, available_attributes, domains, y_train_col_index):
    """Pick an attrubite to split on based on entropy calculations

    Args:
        instances(list of lists): Instance set we're calculating the entropy of
        available_attributes(list): Potential attributes we can split on
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        y_train_col_index(int): Col index of the y_train attribute (not for splitting)
        
    Returns:
        available_attributes[att_to_split_on_index](string): The name of the attribute we're splitting on
    """

    # Calculate the attribute domains dictionary (e.g. standing can be 1 or 2)
    e_news = []
    domains_list = list(domains.items())
    for index in range(len(available_attributes)):
        if y_train_col_index != index: # Skip the att we're trying to predict on (e.g. interviews_well)
            e_news.append(calculate_e_new(index, y_train_col_index, instances, domains_list[index], domains_list[y_train_col_index]))

    # Choose the smallest of the four and split on that, but also check for duplicate e_news calculations (occurs in the interview test dataset)
    try:
        att_to_split_on_index = e_news.index(min(e_news))
        for e_new_index in range(len(e_news)):
            if (e_new_index + 1) < len(e_news):
                if e_news[e_new_index] == e_news[e_new_index + 1] and e_new_index == att_to_split_on_index:
                    att_to_split_on_index = e_new_index + 1
    except ValueError: # For when e_news is empty
        att_to_split_on_index = 0

    return available_attributes[att_to_split_on_index]

def partition_instances(instances, split_attribute, headers, domains):
    """Break up a set of instances into partitions using the split attribute

    Args:
        instances(list of lists): Instance set we're partitioning
        split_attribute(string): Attribute name we're going to be splitting on
        headers(list): Attribute names, corresponds with the instances
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        
    Returns:
        partitions(dict): Partitions organized by attibute value (the key)
    """
    
    # Comments refer to split_attribute "level" in the interview test set
    attribute_domain = domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = headers.index(split_attribute) # 0

    partitions = {} # key (attribute value): value (partition)
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
                
    return partitions

def tdidt(current_instances, available_attributes, headers, domains):
    """Create a tree given a set of instances
    Handles 3 cases (listed in the comments below)

    Args:
        current_instances(list of lists): Instances we're looking at
        available_attributes(list): Attribute names we can still split on
        headers(list): All attribute names
        domains(dict): Possible values for all atts
        
    Returns:
        A constructed tree (as a list of lists of lists...)
    """
    
    # Select an attribute to split on, then remove if from available attributes
    split_attribute = select_attribute(current_instances, available_attributes, domains, (len(available_attributes) - 1))
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]

    # Group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, headers, domains)

    # For each partition, repeat unless one of the following base cases occurs:
    #   CASE 1: All class labels of the partition are the same => make a leaf node
    #   CASE 2: No more attributes to select (clash) => handle clash w/majority vote leaf node
    #   CASE 3: No more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        if len(partition) > 0 and all_same_class(partition): # Case 1
            leaf_subtree = ["Leaf", partition[-1][-1], len(partition), len(current_instances)]
            values_subtree.append(leaf_subtree)
        elif len(partition) > 0 and len(available_attributes) == 0: # Case 2
            leaf_value = perform_majority_voting(current_instances, domains) # Perform majority voting
            # Create a leaf node based on this
            leaf_subtree = ["Leaf", leaf_value, len(partition), len(current_instances)]
            values_subtree.append(leaf_subtree)
        elif len(partition) == 0: # Case 3
            leaf_value = perform_majority_voting(current_instances, domains) # Perform majority voting
            # Create a leaf node based on this
            leaf_subtree = ["Leaf", leaf_value, len(current_instances), len(current_instances)]
            values_subtree.append(leaf_subtree)
        else: # All base cases are false... time to recurse!
            subtree = tdidt(partition, available_attributes.copy(), headers, domains)
            values_subtree.append(subtree)
        tree.append(values_subtree)
    return tree

def perform_majority_voting(clashing_instances, domains):
    """Look for a leaf node value for clashing instances
    Looks for the value that occurs most frequently in the dataset
    If it'd an even split, flip a coin (pick a number 0-the length of the y_train domain)

    Args:
        clashing_instances(list of lists): The instances we're looking for the leaf node value of
        domains(dict): Possible values for each col (e.g. "yes" and "no")
        
    Returns:
       The leaf node value
    """

    is_even_split = True

    # What is our most popular label in this clash?
    domains_list = list(domains.items())
    possible_domain_values = domains_list[-1][1]
    domain_counts = [] # Parallel to possible_domain_values
    for domain_value in possible_domain_values:
        count = 0
        for value in clashing_instances:
            if value[-1] == domain_value:
                count += 1
        domain_counts.append(count)

    # Check if all of the counts are the same (if so, we have an even_split... and if not, find the )
    count_to_check = domain_counts[0]
    for count in domain_counts:
        if count_to_check != count:
            is_even_split = False
            break # Stop searching! We found a difference

    if is_even_split: # Both the same? 50/50? Flip a coin
        coin = random.randint(0, len(possible_domain_values) - 1) # Random number 0-the number of values in y_train domain (e.g. "yes" and "no would be 0-1")
        return possible_domain_values[coin]
    else: # Otherwise, return the value that occurs the most
        index_of_largest_domain_count = domain_counts.index(max(domain_counts))
        return possible_domain_values[index_of_largest_domain_count]

def calculate_entropy(priors, entropy_values):
    """Calculate weighted average of partition entropies
    Priors and entropy values are parallel lists

    Args:
        priors(list): Total occurances in the dataset
        entropy_values(list): Calculated entropy values
        
    Returns:
        avg(int): Average of the priors and entropy values
    """
    
    avg = 0.0
    for i in range(len(entropy_values)):
        avg = avg + (priors[i] * entropy_values[i])
    return avg
    
def calulate_entropy_for_one_partition(values):
    """Calculate the entropy of a partition given values

    Args:
        values(list): The values we're calculating the entropy of
        
    Returns:
        e(float): Calculated entropy
    """

    e = -(values[0] * math.log(values[0], 2))
    index = 1 # Start at index 1 since we've already saved [0] in e
    while index < len(values):
        e = e - (values[index] * math.log(values[index], 2))
        index += 1
    return e
  
def calculate_e_new(col_index, y_train_col_index, instances, domain, y_train_domain):
    """Calculate entropy stats for a domain (priors, entropy for each, total entropy)

    Args:
        col_index(int): Col index of the att we're calulating the entropy of
        y_train_col_index(int): y_train col index
        instances(list of lists): The data table
        domain(dict): Possible values of the att we're calulating the entropy of
        y_train_domain(dict): Possible values of the y_train value
        
    Returns:
        e_new(float): Total entropy
    """
    
    # Find the total number of instances in the dataset
    total = len(instances)
    
    # Load priors (aka how many times do Senior/Mid/Junior appear total?)
    priors = []
    for domain_value in domain[1]: # domain[1] gives a list of domain values
        count = 0
        for instance in instances:
            if instance[col_index] == domain_value:
                count += 1
        priors.append(count/total)
    
    # Entropy of the each domain value (e.g. e of Senior, Mid, and Junior for level)
    # Check for matches (e.g. all cases of Senior and False, then Senior and True...)
    entropy_values = []
    for domain_value in domain[1]:
        values_to_calc = []
        for y_train_domain_value in y_train_domain[1]:
            count = 0
            total = 0
            for instance in instances:
                if instance[col_index] == domain_value:
                    if instance[y_train_col_index] == y_train_domain_value:
                        count += 1 # Both values match! Incremeant the count (numerator)
                    total += 1 # Either way, incremeant the total (denominator)
            if total == 0:
                values_to_calc.append(0.0)
            else:
                values_to_calc.append(count/total)
        
        try:
            e = calulate_entropy_for_one_partition(values_to_calc)
        except ValueError: # For when the calc is undefined
            e = 0.0
        entropy_values.append(e)
        
    # Weighted average of its partition entropies
    e_new = calculate_entropy(priors, entropy_values)
    return e_new

def predict_recursive_helper(tree, X_test):
    """Predict the leaf node based on X_test values
    Handles cases where the y_test attribute is split on

    Args:
        tree(list of lists): The tree we're checking
        X_test(list of lists): Values we're predicting for (subtree that doesn't include the attibute we're predicting)
        
    Returns:
        Either the leaf value or a recursive call to this function
    """
    
    label = tree[0] # e.g. Attribute, Value, or Leaf
    if label == "Attribute":
        # Get the index of this attribute using the name (i.e. att0 is at index [0] in the attribute names)
        att_index = 0 # Default is 0
        for letter in tree[1]:
            if letter != "a" and letter != "t":
                att_index = letter
        att_index = int(att_index)
        
        # In case we split on the class label
        if att_index >= len(X_test):
            return tree[2][2][1]
        
        # Grab the value at this index and see if we have a match going down the tree
        instance_value = X_test[att_index]
        i = 2
        while i < len(tree):
            value_list = tree[i]
            if value_list[1] == instance_value: # Recurse when a match is found
                return predict_recursive_helper(value_list[2], X_test)
            i += 1
    else:
        return tree[1] # Grab the value of the leaf

def make_rule_for_one_branch(tree, attribute_names, class_name, rule):
    """Grab a list of strings that represents one branch's rule (see args for formatting)
    Assumes that ruleis already populated with the split attribue info (["IF", "att0", "=", "value"]) upon initial call
    
    Args:
        tree(list of lists): The tree/subtree we're looking at
        attribute_names(list of str or None): A list of attribute names to use in the decision rules
            (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
        class_name(str): A string to use for the class name in the decision rules
            ("class" if a string is not provided and the default name "class" should be used).
        rule(list of strings): A rule, formatted like IF att0 == value AND ... THEN class_label = True (but with each element in an encapsulating list)

    Returns:
        list of strings: The full branch's rule
    """
    
    label = tree[0] # e.g. Attribute, Value, or Leaf
    if label == "Leaf": # We've hit the end of a branch
        rule.append("THEN")
        rule.append(class_name)
        rule.append("==")
        rule.append(tree[1]) # The value of the leaf
        return rule
    elif label == "Attribute":
        rule.append("AND")
        if attribute_names == None:
            rule.append(tree[1])
        else: # [-1]st spot of att labels is the index
            att_index = int(tree[1][-1])
            rule.append(attribute_names[att_index])
        rule.append("==")

        # There will be more to the initial tree beyond the leaf we run into here, because of this attribute split (e.g. one rule where phd = yes and one where phd = no)
        index = 2 # Values start at index 2
        new_rules = []
        
        while index < len(tree): # Go through the values on each partition (e.g. Junior, Mid and Senior)
            # Calculate the branch (initial attribute that's already in there is passed in as rule)
            new_rule = make_rule_for_one_branch(tree[index], attribute_names, class_name, rule)

            new_rules.append(new_rule)
            index += 1
            if index < len(tree): # Check if we've hit the end of the tree (and if so, don't add any more rules)
                rule = []
                if attribute_names == None:
                    rule = [tree[1], "=="]
                else: # [-1]st spot of att labels is the index
                    att_index = int(tree[1][-1])
                    rule = [attribute_names[att_index], "=="]

        return new_rules
    else: # Otherwise, it's a value
        rule.append(tree[1])
        return make_rule_for_one_branch(tree[2], attribute_names, class_name, rule) # Recurse on subtree