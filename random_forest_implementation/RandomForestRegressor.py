from random_forest_implementation.regression_tree import regression_tree
from random_forest_implementation.bootstrap_sampling import bootstrap_sampling

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
            print("creating bootstrapped sample.")
            bootstrapped_dataset = bootstrap_sampling(dataset)
            print("done.")

            print("creating regression tree.")
            reg_tree_root = regression_tree(bootstrapped_dataset, self.min_samples, self.max_depth, 0)
            print("done.")

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
    