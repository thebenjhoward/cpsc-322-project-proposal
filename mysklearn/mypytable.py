##############################################
# Programmer: Elizabeth Larson (starter code by Dr. Gina Sprint)
# Class: CPSC 322-01, Spring 2021
# Programming Assignment #6
# 04/14/21
# I didn't attempt the bonus. TODO: Edit (or not)
#
# Sources:
#   Using remove(): https://note.nkmk.me/en/python-list-clear-pop-remove-del/
#   Creating an empty list with a fixed size: https://stackoverflow.com/questions/10712002/create-an-empty-list-in-python-with-certain-size
#   Working with the csv module: https://docs.python.org/3/library/csv.html
# 
# Description: This program handles the logic for creating a MyPyTable object.
#              It handles loading and saving to a CSV file, generating stats,
#              changing data (e.g. work with missing values and duplicate rows),
#              and working with joins.
##############################################


import copy
import csv 
import mysklearn.myutils as myutils
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """

        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """

        N = len(self.data) # Number of rows (checks the whole table)
        M = len(self.data[0]) # Number of cols (only checks the first row)

        return N, M

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col = []
        col_index = 0

        if len(self.data) > 0: # Doesn't grab the col info if the dataset is empty
            if type(col_identifier) == int: # If we're looking for a col index...
                if col_identifier >= len(self.data) or col_identifier < 0:
                    raise ValueError
                else:
                    col_index = col_identifier
            elif type(col_identifier) == str: # Otherwise, it's a string... find the index
                col_index = self.column_names.index(col_identifier)

            for i in range(len(self.data)):
                if self.data[i][col_index] == "NA": # Check if there's a missing value in the col
                    include_missing_values = True
                else:
                    include_missing_values = False
                col.append(self.data[i][col_index]) # Keep track of the value at this spot in the col       

        return col

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                try: # Try to convert to a float and save in dataset if possible
                    float_value = float(self.data[i][j])
                    self.data[i][j] = float_value
                except ValueError: # Otherwise, skip it
                    pass
        
    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        
        for i in range(len(rows_to_drop)):
            try:
                self.data.remove(rows_to_drop[i])
            except ValueError:
                pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """

        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            self.column_names = next(reader) # Take note of the header names and save them seperate from the data
            for row in reader: # Now save the data row by row
                self.data.append(row)

        self.convert_to_numeric() # Convert values to floats (that can be converted)

        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """

        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names) # Write col names
            writer.writerows(self.data) # Write data

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """

        duplicate_rows = []
        checked_keys = []
        
        for r in range(len(self.data)):
            # Keep track of the keys in each row
            new_checked_key = []
            key_index = 0
            for c in range(len(self.data[0])):
                if key_index < len(key_column_names):
                    if self.column_names[c] == key_column_names[key_index]: # The col we are at is in the key
                        new_checked_key.append(self.data[r][c])
                        key_index += 1
                else: # The col we are at is NOT in the key... skip it
                    c += 1
            
            # Check if we've seen this key in another row before
            if not new_checked_key in checked_keys:
                checked_keys.append(new_checked_key)
            else: # If it's a duplicate, keep track of that in duplicate_rows
                duplicate_rows.append(self.data[r])

        return duplicate_rows

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """

        rows_to_drop = []

        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == "NA":
                    rows_to_drop.append(self.data[i]) # If the row has an empty value, add it to the running list of rows to drop
        
        self.drop_rows(rows_to_drop) # Drop all of the rows we found that have missing values

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """

        # Search the data set for missing values
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] == "NA": # If a missing value has been found...
                    col_contents = self.get_column(col_name) # Grab the contents of the desired col

                    # Find the average of the col
                    col_sum = 0
                    for k in range(len(col_contents)):
                        if type(col_contents[k]) == float:
                            col_sum += col_contents[k]
                    col_avg = col_sum / (len(col_contents) - 1) # 1 for skipping the missing value in this len() count

                    self.data[i][j] = col_avg # Replace NA index with col average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """

        # Create a table for stats and label it with a col names
        stats_table = MyPyTable()
        stats_table.column_names = ["attribute", "min", "max", "mid", "avg", "median"]

        for col_name in col_names:
            col_info = self.get_column(col_name) # Grab the col values for the desired col
            if col_info != []: # Check if col is empty (i.e. data is [])
                row = [] # Represents a row in the stats_table

                row.append(col_name) # Attribute (col name)
                row.append(min(col_info)) # Max value
                row.append(max(col_info)) # Min value
                row.append((min(col_info) + max(col_info)) / 2) # Mid value

                # Average
                col_sum = 0
                for i in range(len(col_info)):
                    col_sum += col_info[i]
                row.append(col_sum / len(col_info))

                # Median
                sorted_col_info = col_info.copy()
                sorted_col_info.sort()
                if len(sorted_col_info) % 2 == 0: # Even number of entries in sorted_col_info
                    median1 = sorted_col_info[len(sorted_col_info) // 2]
                    median2 = sorted_col_info[(len(sorted_col_info) // 2) - 1]
                    row.append((median1 + median2) / 2)
                else: # Odd number of enteries in sorted_col_info
                    row.append(sorted_col_info[len(sorted_col_info) // 2])
                
                stats_table.data.append(row) # Add the finished product to the stats table

        return stats_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """

        inner_join_table = MyPyTable()

        # Keep track of self.data headers (no repeats... i.e. keys)
        for i in range(len(self.column_names)):
            if not self.column_names[i] in inner_join_table.column_names:
                inner_join_table.column_names.append(self.column_names[i])

        # Keep track of other_table headers (no repeats... i.e. keys)
        for i in range(len(other_table.column_names)):
            if not other_table.column_names[i] in inner_join_table.column_names:
                inner_join_table.column_names.append(other_table.column_names[i])

        is_match = False # True if values in a given row match on all key cols
        for x in self.data:
            for y in other_table.data:
                for key in key_column_names: # Go through all keys in the list
                    if x[self.column_names.index(key)] == y[other_table.column_names.index(key)]:  # If there's a match on the key values...
                        is_match = True
                    else: # Otherwise, it's NOT a match on key... stop the loop
                        is_match = False
                        break
                if is_match == True:
                    new_row = [] # Start building a new row of info
                    for value1 in x:
                        new_row.append(value1)
                    for value2 in y:
                        if not value2 in new_row:
                            new_row.append(value2)
                    inner_join_table.data.append(new_row)
                        
        return inner_join_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """

        outer_join_table = MyPyTable()

        # Keep track of self.data headers (no repeats... i.e. keys)
        for i in range(len(self.column_names)):
            if not self.column_names[i] in outer_join_table.column_names:
                outer_join_table.column_names.append(self.column_names[i])

        # Keep track of other_table headers (no repeats... i.e. keys)
        for i in range(len(other_table.column_names)):
            if not other_table.column_names[i] in outer_join_table.column_names:
                outer_join_table.column_names.append(other_table.column_names[i])

        # Because we'll be deleting some values, make copies of the data so the original source isn't messed with
        table1_copy = self.data.copy()
        table2_copy = other_table.data.copy()

        is_match = False # True if values in a given row match on all key cols
        for x in self.data:
            for y in other_table.data:
                for key in key_column_names: # Go through all keys in the list
                    if x[self.column_names.index(key)] == y[other_table.column_names.index(key)]:  # If there's a match on the key values...
                        is_match = True
                    else: # Otherwise, it's NOT a match on key... stop the loop
                        is_match = False
                        break
                if is_match == True:
                    new_row = [] # Start building a new row of info
                    for value1 in x:
                        new_row.append(value1)
                    for value2 in y:
                        if not value2 in new_row:
                            new_row.append(value2)
                    outer_join_table.data.append(new_row)
                    
                    # Remove checked key values from the copy tables (with try/except for arrays of different sizes)
                    try:
                        table1_copy.remove(x)
                    except ValueError:
                        pass
                    try:
                        table2_copy.remove(y)
                    except ValueError:
                        pass

        # Case 2: Missing values in the 1st table
        for row in table1_copy:
            new_row = []
            
            # Make row of Nones (to be filled later)
            for i in range(len(outer_join_table.column_names)):
                new_row.append(None)

            # Add values from the 1st table to the joined table in their corresponding spots
            for col in range(len(self.column_names)):
                for header_col in range(len(outer_join_table.column_names)):
                    if outer_join_table.column_names[header_col] == self.column_names[col]:
                        new_row[header_col] = row[col]
                    
            # If None, fill with "NA"
            for col in range(len(new_row)):
                if new_row[col] == None:
                    new_row[col] = "NA"
            
            outer_join_table.data.append(new_row)
            
        # Case 3: Missing values in the 2nd table
        for row in table2_copy:
            new_row = []
            
            # Make row of Nones (to be filled later)
            for i in range(len(outer_join_table.column_names)):
                new_row.append(None)

            # Add values from the 1st table to the joined table in their corresponding spots
            for col in range(len(other_table.column_names)):
                for header_col in range(len(outer_join_table.column_names)):
                    if outer_join_table.column_names[header_col] == other_table.column_names[col]:
                        new_row[header_col] = row[col]
                    
            # If None, fill with "NA"
            for col in range(len(new_row)):
                if new_row[col] == None:
                    new_row[col] = "NA"
            
            outer_join_table.data.append(new_row)

        return outer_join_table