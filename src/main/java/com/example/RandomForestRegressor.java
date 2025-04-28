package com.example;

import java.util.List;
import smile.data.DataFrame;
import com.example.TreeNode;

public class RandomForestRegressor {
    // Attributes
    int nEstimators;
    int minSamples;
    int maxDepth;
    boolean isTrained;
    List<TreeNode> forest;

    // Constructor
    public RandomForestRegressor(int nEstimators, int minSamples, int maxDepth) {
        this.nEstimators = nEstimators;
        this.minSamples = minSamples;
        this.maxDepth = maxDepth;
        this.isTrained = false;
    }

    // Methods
    public void fit(DataFrame df) {
        // Loop over the amount of nEstimators
            // a. Create a bootstrapped dataset, calling bootstrapSample()
            // b. Create a regression tree on that dataset, calling createRegressionTree()
            // c. Save the regression tree in the forest
    }
    public float predict(DataFrame df) {
        // 1. Initialize an empty predictedScores list
        // 2. Loop over every regression tree in the forest
            // a. While you haven't reached a leaf node, continue traversing the regression tree
            // b. Once a leaf node is reached, add its value to the predictedScores list
        // 3. Grab the average of all predicted scores and return it
        return 0.0f;
    }
    public float decay(int daysSinceLastLoggedIn) {
        // 1. If parameter is >= 30, return 0.10
        // 2. Else If parameter btwn [3,30], return 1 - log_base_10(parameter) + 0.477
        // 3. Else, return 1.00 (no decay)
        return 0.0f;
    }
    public TreeNode createRegressionTree(DataFrame df, int minSamples, int maxDepth, int currentDepth) {
        // 1. Grab the labels that determine the target variable and the target label itself

        // 2. Keep track of the best label, its threshold, and its SSR (sum of squared residuals)

        // 3. Loop over every label (excluding target label)
            // a. Keep track of this labels best threshold and its SSR
            // b. Loop over all rows of the dataframe
                // i. Grab current and next row
                // ii. Calculate current threshold (average of both rows for current label)
                // iii. Filter the dataset, grabbing all rows whose current label is less than current threshold
                // iv. Grab the average target label value from those rows
                // v. append a new column to iii., calculating the squared residual of each row
                // vi. repeate steps iii-v for rows whose current label is greater than or equal to the current threshold
                // vii. sum left and right squared residuals from the threshold into a combined sum
                // vii. update label best threshold if it has a SSR lower than the current best threshold
            // c. update global best label if this current label has a lower SSR value
        
        // 4. Now, the best candidate is chose for the root node. 
        // Create the root node, passing in a string containing the global best label and its threshold to the rule

        // 5. Filter the dataframe into rows less than and rows greater than or equal to the global threshold

        // 6. Conditionally check to crete a leaf node or continue splitting based on maxDepth and minSamples
        return new TreeNode("");
    }
    public DataFrame bootstrapSample(DataFrame df) {
        // 1. Grab the number of rows, n

        // 2. Make an empty list to store random indices to be part of the bootstrapped dataset

        // 3. Loop n times
            // a. Generate a random index, add to the list of indices

        // 4. Create a new DataFrame passing in the list of random indices

        // 5. Return that dataframe
        return DataFrame.of(new int[][]{{0,0}, {0,0}});
    }
}
