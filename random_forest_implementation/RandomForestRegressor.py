from regression_tree import regression_tree

class RandomForestRegressor:
    def __init__(self, min_samples, n_estimators):
        self.min_samples = min_samples
        self.n_estimators = n_estimators
    
    def train(self, dataset):
        # 1. Loop over the amount of n_estimators
            # a. Create a bootstrapped dataset
            # b. Create a regression tree on that bootstrapped dataset
            # c. Save the regression tree in a forest (data structure for the trees)
        ...

    def predict(self, inputs):
        # 1. Initialize an empty predicted_scores list
        # 2. Loop over every regression tree in the forest
            # a. Traverse the current tree according to values of inputs
            # b. Once a leaf node is reached, add its value to the predicted_scores list
        # 3. Sum the predicted scores in the list and divide by n_estimators
        # 4. Return the value obtained in step 3
        ...