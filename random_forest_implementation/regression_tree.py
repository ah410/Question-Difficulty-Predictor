from random_forest_implementation.tree_node import TreeNode

def regression_tree(dataset, min_samples, max_depth, current_depth):
    # Grab the labels need for both input variables and the target label
    labels = list(dataset.keys())
    target_label = labels.pop()

    # Keep track of the best label with a threshold that minimizes SSR
    global_best_sum_of_squared_residuals = float('inf')
    global_best_threshold = None
    global_best_label = None

    for label in labels:
        # Keep track of the current label's best threshold and SSR values
        label_best_sum_of_squared_residuals = float('inf')
        label_best_threshold = None

        for i in range(len(dataset) - 1):
            row = dataset.iloc[i]
            next_row = dataset.iloc[i+1]

            current_threshold = (row[label] + next_row[label]) / 2

            # Filter for rows less than the current threshold and find the average value of current label
            less_than = dataset.query(f'{label} < {current_threshold}').copy()
            left_average = less_than.mean()[target_label]
            less_than['squared_residual'] = less_than.apply(lambda row: pow(row[target_label] - left_average, 2), axis=1)

            # Filter for rows greater than or equal to the current threshold and find the average value of current label
            greater_than = dataset.query(f'{label} >= {current_threshold}').copy()
            right_average = greater_than.mean()[target_label]
            greater_than['squared_residual'] = greater_than.apply(lambda row: pow(row[target_label] - right_average, 2), axis=1)

            # Add up both left and right SSRs to get the current SSR for the current threshold of this label
            current_sum_of_squared_residuals = less_than.sum()['squared_residual'] + greater_than.sum()['squared_residual']

            # Update the current label's best threshold value that minimizes the SSR value
            if current_sum_of_squared_residuals < label_best_sum_of_squared_residuals:
                label_best_sum_of_squared_residuals = current_sum_of_squared_residuals
                label_best_threshold = current_threshold

        # Found the best threshold for this label. Compare against the best label found so far
        if label_best_sum_of_squared_residuals < global_best_sum_of_squared_residuals:
            global_best_sum_of_squared_residuals = label_best_sum_of_squared_residuals
            global_best_threshold = label_best_threshold
            global_best_label = label

    # Create the root node for the regression tree
    root = TreeNode(rule=f"{global_best_label} < {global_best_threshold}")
        
    # Grab the number of datapoints(rows) that have labels < and >= the threshold for that label
    less_than_datapoints = dataset.query(f'{global_best_label} < {global_best_threshold}')
    greater_than_datapoints = dataset.query(f'{global_best_label} >= {global_best_threshold}')

    # Create a leaf node if rows are less than the minimum number of samples required or exceeded the max depth
    # Else, continue splitting for the left sub-tree 
    if len(less_than_datapoints) < min_samples or current_depth >= max_depth:
        average = less_than_datapoints.mean()[target_label]
        left_child = TreeNode(prediction=average)
        root.left = left_child
    else:
        root.left = regression_tree(less_than_datapoints, 5, max_depth, current_depth+1)

    # Create a leaf node if rows are less than the minimum number of samples required or exceeded the max depth
    # Else, continue splitting for the right sub-tree 
    if len(greater_than_datapoints) < min_samples or current_depth >= max_depth:
        average = greater_than_datapoints.mean()[target_label]
        right_child = TreeNode(prediction=average)
        root.right = right_child
    else:
        root.right = regression_tree(greater_than_datapoints, 5, max_depth, current_depth+1)
    
    return root


def print_bst(node, level=0):
    if node != None:
        print_bst(node.left, level + 1)
        if node.rule != "":
            print(' ' * 4 * level + '-> ' + str(node.rule))
        else:
            print(' ' * 4 * level + '-> ' + str(node.prediction))
        print_bst(node.right, level + 1)
