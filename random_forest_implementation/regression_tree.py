from tree_node import TreeNode

def createRegressionTree(dataset):
    # Start with the first column variable, Age
        # Try different threshold values of Age and calculate the sum of squared residual for the current threshold value.
            # Each threshold value will be found by looping over the rows, for every 2 rows grab the Ages and compute average Age.
            # Find the average age values for all observations less than the average of the 2 chosen points and greater than. These will be left and right leaves.
            # For both left & right leaves, calculate each observations residual by taking actual Age - predicted Age (average). Then square them.
            # Sum of squared residuals found by adding up all your calculated squared residuals. You can plot these on a graph where x-axis = variable threshold and y-axis = sum of squared residuls
                # to visually views which threshold value has the smallest sum of squared residuals.
            # Keep track of the threshold value with the smallest sum of squared residual, resets after you move to next column.
        # After best threshold value is found for Age, create the root node with the rule "Age < {threshold_value}"
        # Find the number of observations(rows) where Age is less than and greater than threshold
        # If number of observations less than threshold of Age is < N (tbd), stop splitting and create a left leaf node with the prediction being the average target variable of all remaining observations.
        # Else, continue splitting or subdividing the observation of Age
        # If number of observations greater than threshold of Age is < N (tbd), stop splitting and create a left leaf node with the prediction being the average target variable of all remaining observations.
    # Do the same for the rest of the columns, adding to the regression tree
    
    return ""