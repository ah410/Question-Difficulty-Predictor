from tree_node import TreeNode
import pandas as pd

def createRegressionTree(dataset):
    # Test dataset on simple data, will convert to student dataset after regression trees are confirmed working
    labels = ['Age', 'Dosage']
    min_samples = 5
    global_best_sum_of_squared_residuals = float('inf')
    global_best_threshold = None
    global_best_label = None

    for label in labels:
        current_subset = dataset[[label, 'Effectiveness']].sort_values(label)
        label_best_sum_of_squared_residuals = float('inf')
        label_best_threshold = None

        # Loop over all rows
        for i in range(len(current_subset) - 1):
            row = current_subset.iloc[i]
            next_row = current_subset.iloc[i+1]

            current_threshold = (row[label] + next_row[label]) / 2

            # Get list of all labels less than threshold, compute leftAverage for effectiveness
            # For leftAverage effectiveness, find the squared residual for each datapoint, where the predicted effectiveness is the leftAverage effectiveness computed above
            less_than = current_subset.query(f'{label} < {current_threshold}').copy()
            left_average = less_than.mean()['Effectiveness']
            less_than['squared_residual'] = less_than.apply(lambda row: pow(row['Effectiveness'] - left_average, 2), axis=1)

            # Get list of all labels greater than threshold, compute rightAverage for effectiveness
            # For rightAverage effectiveness, find the squared residual for each datapoint, where the predicted effectiveness is the rightAveraverage effectiveness computed above
            greater_than = current_subset.query(f'{label} >= {current_threshold}').copy()
            right_average = greater_than.mean()['Effectiveness']
            greater_than['squared_residual'] = greater_than.apply(lambda row: pow(row['Effectiveness'] - right_average, 2), axis=1)

            # The current threshold's sum of squared residuals for this label.
            current_sum_of_squared_residuals = less_than.sum()['squared_residual'] + greater_than.sum()['squared_residual']

            # Update the current labels threshold value that minimizes the sum of squared residuals
            if current_sum_of_squared_residuals < label_best_sum_of_squared_residuals:
                label_best_sum_of_squared_residuals = current_sum_of_squared_residuals
                label_best_threshold = current_threshold

        # Found the best threshold for this label. Compare against the best label found so far.
        if label_best_sum_of_squared_residuals < global_best_sum_of_squared_residuals:
            global_best_sum_of_squared_residuals = label_best_sum_of_squared_residuals
            global_best_threshold = label_best_threshold
            global_best_label = label

    # Create the root node for the regression tree
    root = TreeNode(rule=f"{global_best_label} < {global_best_threshold}")
        
    # Grab the number of observations for this label's threshold less than and greater than it
    less_than_datapoints = dataset.query(f'{global_best_label} < {global_best_threshold}')
    greater_than_datapoints = dataset.query(f'{global_best_label} >= {global_best_threshold}')

    # If-Else statements to conditionally check if the number of observations meets the minimum 
    if len(less_than_datapoints) < min_samples:
        # Calculate the average target variable value
        average = less_than_datapoints.mean()['Effectiveness']

        # Set as a leaf node for the left child of root
        left_child = TreeNode(prediction=average)
        root.left = left_child
    else:
        # Continue splitting the data for the left subtree
        root.left = createRegressionTree(less_than_datapoints)
    
    if len(greater_than_datapoints) < min_samples:
        # Calculate the average target variable value
        average = greater_than_datapoints.mean()['Effectiveness']

        # Set as a leaf node for the right child of root
        right_child = TreeNode(prediction=average)
        root.right = right_child
    else:
        # Continue splitting the data for the right subtree
        root.right = createRegressionTree(greater_than_datapoints)
        
    return root


def print_bst(node, level=0):
    if node != None:
        print_bst(node.left, level + 1)
        if node.rule != "":
            print(' ' * 4 * level + '-> ' + str(node.rule))
        else:
            print(' ' * 4 * level + '-> ' + str(node.prediction))
        print_bst(node.right, level + 1)

dataset = pd.read_csv('./dataset/dosage_effectiveness.csv')
root_node = createRegressionTree(dataset)
print_bst(root_node)