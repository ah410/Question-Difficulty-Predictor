import random

from tree_node import TreeNode


class RandomForestRegressor2:
    def __init__(self, n_estimators, min_samples, max_depth):
        self.forest = []
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.max_depth = max_depth
    
    def fit(self, dataset):
        # 1. Loop over the amount of n_estimators
            # a. Create a bootstrapped dataset
            # b. Create a regression tree on that bootstrapped dataset
            # c. Save the regression tree in a forest (data structure for the trees)
        for i in range(self.n_estimators):
            print("creating bootstrapped sample...")
            bootstrapped_dataset = self.bootstrap_sampling(dataset)

            print("creating regression tree...")
            reg_tree_root = self.regression_tree(bootstrapped_dataset, self.min_samples, self.max_depth, 0)

            self.forest.append(reg_tree_root)
        
        return None

    def predict(self, inputs):
        # inputs = {"label1": value1, "label2": value2, ...}
        # 1. Initialize an empty predicted_scores list
        # 2. Loop over every regression tree in the forest
            # a. Traverse the current tree according to values of inputs
            # b. Once a leaf node is reached, add its value to the predicted_scores list
        # 3. Sum the predicted scores in the list and divide by n_estimators
        # 4. Return the value obtained in step 3
        predicted_scores = []

        for _, tree_node in enumerate(self.forest):
            current_node = tree_node

            # Continue exploring the tree until you've reached an empty rule (leaf node)
            while current_node.rule != "":
                # (rule syntax: "label < some_value")
                split_rule = current_node.rule.split()
                label = split_rule[0]
                value = split_rule[2]

                if inputs[label].values[0] < float(value) and current_node.left:
                    current_node = current_node.left
                elif inputs[label].values[0] >= float(value) and current_node.right:
                    current_node = current_node.right

            # Reached a leaf node, add its prediction to the list
            predicted_scores.append(current_node.prediction)

        return sum(predicted_scores)/len(predicted_scores)

    def bootstrap_sampling(self, dataset):
        # 1. Grab number of rows of original dataset
        n = len(dataset)

        # 2. Make an empty list to store the random indices for all rows to be used in the bootstrapped dataset
        random_indices = []

        # 3. For size N of original dataset
            # a. Generate a random index in range [0,N-1] inclusive and add to random_indices list
        for _ in range(n):
            random_indices.append(random.randint(0, n - 1))

        # 4. Create a new DataFrame passing in the random_indices list
        bootstrapped_dataset = dataset.iloc[random_indices]

        # 5. Return that dataframe
        return bootstrapped_dataset

    def regression_tree(self, dataset, min_samples, max_depth, current_depth):
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
            root.left = self.regression_tree(less_than_datapoints, 5, max_depth, current_depth+1)

        # Create a leaf node if rows are less than the minimum number of samples required or exceeded the max depth
        # Else, continue splitting for the right sub-tree 
        if len(greater_than_datapoints) < min_samples or current_depth >= max_depth:
            average = greater_than_datapoints.mean()[target_label]
            right_child = TreeNode(prediction=average)
            root.right = right_child
        else:
            root.right = self.regression_tree(greater_than_datapoints, 5, max_depth, current_depth+1)
        
        return root
