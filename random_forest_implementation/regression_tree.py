from tree_node import TreeNode
import pandas as pd

def createRegressionTree(dataset):
    # Initialize a variable to store the column and its sum of squared residuals value that is the lowest of all columns
    # Initialize min_samples to stop splitting 

    # For each column in your dataset that influences the target variable (e.g. Age, Sex, Dosage):
        # Initialize a variable to hold the best threshold value for that column, best_threshold, to None
        # Initialize a variable to hold the sum of squared residuals that is the minimum of all found, to positive infinity
        # For each row in your dataset specific to that column up to the second to last row:
            # Initialize a variable to hold the current sum of squared residuals, to postivie infinity
            # Grab the current row and next row
            # Calculate the current threshold by taking the average of both rows, store in current_threshold
            # Find the average target variable less than the threshold by summing up all the datapoints target variable / number of datapoints (left)
            # Find the average target variable greater than the threshold by summing up all the datapoints target variable / number of datapoints (right)
            # For all datapoints less than the current_theshold:
                # Compute a squared residual for that datapoint | squared residual = (actual target variable - left_average)^2
                # Add it to the current sum of squared residuals variable
            # For all datapoints greater than the average of both rows:
                # Compute a squared residual for that datapoint | squared residual = (actual target variable - right_average)^2
                # Add it to the current sum of squared residuals variable
        # Compare the sum of squared residuals with the current best. If less than, replace. Else, keep the same.
    
    # Now you have the best column. Make it the root node, passing in the appropriate rule in the node creation
    # Remove that column from the dataset
    # If number of observations less than the threshold is less than min_samples
        # Calculate average prediction
        # Create leaf node with the average prediction
    # Else:
        # Continue splitting with the updated dataset (Recursive Partitioning)

    # If number of observations greater than the threshold is less than min_samples
        # Calculate average prediction
        # Create leaf node with the average prediction
    # Else:
        # Continue splitting with the updated dataset. (Recursive Partitioning)

    # Test dataset on simple data, will convert to student dataset after regression trees are confirmed working
    labels = ['Age', 'Dosage']
    min_samples = 5
    best_column = {}

    # Loop over all labels that determine the target variable
    for label in labels:
        current_subset = dataset[[label, 'Effectiveness']].sort_values(label)
        best_threshold = None
        best_sum_of_squared_residuals = float('inf')

        # Loop over all rows
        for i in range(len(current_subset) - 1):
            current_sum_of_squared_residuals = float('inf')
            row = current_subset.iloc[i]
            next_row = current_subset.iloc[i+1]

            current_threshold = (row[label] + next_row[label]) / 2

            # Get list of all labels less than threshold, compute leftAverage for effectiveness
            # For leftAverage effectiveness, find the squared residual for each datapoint, where the predicted effectiveness is the leftAverage effectiveness computed above
            less_than = current_subset.query(f'{label} < {current_threshold}').copy()
            left_average = less_than.mean()['Effectiveness']
            less_than['squared_residual'] = less_than.apply(lambda row: pow(row['Effectiveness'] - left_average, 2), axis=1)

            print("current threshold value: ", current_threshold)
            print(f"observations less than threshold: {len(less_than)}")
            print(f"left average dose effectiveness: {left_average}")


            # Get list of all labels greater than threshold, compute rightAverage for effectiveness
            # For rightAverage effectiveness, find the squared residual for each datapoint, where the predicted effectiveness is the rightAveraverage effectiveness computed above
            greater_than = current_subset.query(f'{label} >= {current_threshold}').copy()
            right_average = greater_than.mean()['Effectiveness']
            greater_than['squared_residual'] = greater_than.apply(lambda row: pow(row['Effectiveness'] - right_average, 2), axis=1)

            print(f"observations greater than threshold: {len(greater_than)}")
            print(f"right average dose effectiveness: {right_average}")

            print(less_than.head(1))
            print(greater_than.head(1))
            print()

            # Add up all squared residuals to get one final number. This is the number you want to minimize.

            # Check if this sum of squared residuals is better than the current best, replace if so. Else, bestThreshold stays the same.
        
        # Found the bestThreshold for this column. Create your node and add the rule

        # If the number of observations for less than and greater than the threshold are more than min_samples, continue splitting. Else, stop splitting
            # Likely be some recursion implementation. Continue calling this function, passing in the reduced dataset you need. returns a Node for your regression tree with children

    # Need to find out which column has the lowest sum of squared residuals. that will be the root node.

    return ""


dataset = pd.read_csv('./dataset/dosage_effectiveness.csv')
createRegressionTree(dataset)
