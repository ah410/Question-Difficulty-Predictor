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

    return ""
