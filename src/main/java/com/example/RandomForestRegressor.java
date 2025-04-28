package com.example;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalDouble;

import smile.data.DataFrame;
import smile.data.vector.ValueVector;
import smile.util.Index;

public class RandomForestRegressor {
    // Attributes
    int nEstimator;
    int minSamples;
    int maxDepth;
    boolean isTrained;
    List<TreeNode> forest;

    // Constructor
    public RandomForestRegressor(int nEstimator, int minSamples, int maxDepth) {
        this.nEstimator = nEstimator;
        this.minSamples = minSamples;
        this.maxDepth = maxDepth;
        this.isTrained = false;
    }

    // Methods
    public void fit(DataFrame df) {
        // Create nEstimator trees where each tree is trained on a bootstrapped dataset with replacement
        for (int i = 0; i < nEstimator; i++) {
            DataFrame bootstrappedDf = bootstrapSample(df);
            TreeNode root = createRegressionTree(bootstrappedDf, minSamples, maxDepth, 0);
            forest.add(root);
        }
    }
    public float predict(DataFrame df) {
        // 1. Initialize an empty predictedScores list
        List<Float> predictedScores = new ArrayList<>();

        // 2. Loop over every regression tree in the forest
        for (TreeNode node : forest) {
            TreeNode current = node;
            // a. While you haven't reached a leaf node, continue traversing the regression tree
            while (!"".equals(current.getRule())) {
                // current.rule syntax: "label < value"
                String[] ruleSplit = current.getRule().split(" ");
                String ruleLabel = ruleSplit[0];
                String ruleValue = ruleSplit[2];

                ValueVector columns = df.column(ruleLabel);
                float inputValue = columns.getFloat(0);
                
                if (inputValue < Float.parseFloat(ruleValue) && current.getLeftChild() != null) {
                    current = current.getLeftChild();
                } 
                else if (inputValue >= Float.parseFloat(ruleValue) && current.getRightChild() != null) {
                    current = current.getRightChild();
                }
            }

            // b. Once a leaf node is reached, add its value to the predictedScores list
            predictedScores.add(current.getPrediction());
        }
        
        // 3. Grab the average of all predicted scores and return it
        OptionalDouble average = predictedScores.stream().mapToDouble(Float::doubleValue).average();

        if (average.isPresent()) {
            return (float) average.getAsDouble();
        } else {
            return 0.0f;
        }
    }
    public float decay(int daysSinceLastLoggedIn) {
        // 1. If parameter is >= 30, return 0.10
        if (daysSinceLastLoggedIn >= 30) {
            return 0.10f;
        }
        // 2. Else If parameter btwn [3,30], return 1 - log_base_10(parameter) + 0.477
        else if (daysSinceLastLoggedIn > 3) {
            return (float) (1 - Math.log10(daysSinceLastLoggedIn) + 0.477);
        } 
        // 3. Else, return 1.00 (no decay)
        else {
            return 1.00f;
        }
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
        int n = df.nrow();

        // 2. Make an empty list to store random indices to be part of the bootstrapped dataset
        List<Integer> randomIndices = new ArrayList<>();

        // 3. Loop n times
            // a. Generate a random index, add to the list of indices
        for (int i = 0; i < n; i++) {
            int randomIndex = (int) (Math.random() * n); 
            randomIndices.add((randomIndex));
        }

        // 4. Create a new DataFrame passing in the list of random indices
        int[] indicesArray = randomIndices.stream().mapToInt(Integer::intValue).toArray();
        Index index = Index.of(indicesArray);
        DataFrame newDf = df.apply(index);

        // 5. Return that dataframe
        return newDf;
    }
}
