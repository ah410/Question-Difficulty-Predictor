package com.example;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.stream.Stream;

import smile.data.Collectors;
import smile.data.DataFrame;
import smile.data.Row;
import smile.data.Tuple;
import smile.data.vector.ValueVector;
import smile.util.Index;

public class RandomForestRegressor {
    // Attributes
    int nEstimator;
    int minSamples;
    int maxDepth;
    boolean isTrained;
    List<TreeNode> forest = new ArrayList<>();

    // Constructor
    public RandomForestRegressor(int nEstimator, int minSamples, int maxDepth) {
        this.nEstimator = nEstimator;
        this.minSamples = minSamples;
        this.maxDepth = maxDepth;
        this.isTrained = false;
    }

    // Methods
    public void fit(DataFrame df) {
        // 1. Filter DataFrame to only grab the columns ["correctLast3","avgTime","confidence","quizScore","sessions","readinessScore"]
        DataFrame filteredDf = df.apply("correctLast3","avgTime","confidence","quizScore","sessions","readinessScore");       

        // 2. Create nEstimator trees where each tree is trained on a bootstrapped dataset with replacement
        for (int i = 0; i < nEstimator; i++) {
            DataFrame bootstrappedDf = bootstrapSample(filteredDf);
            TreeNode root = createRegressionTree(bootstrappedDf, minSamples, maxDepth, 0);
            forest.add(root);
        }
    }
    public float predict(Row row) {
        // 1. Initialize an empty predictedScores list to store each regression trees score
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

                float inputValue = row.getFloat(ruleLabel);
                
                if (inputValue < Float.parseFloat(ruleValue) && current.getLeftChild() != null) {
                    current = current.getLeftChild();
                } 
                else if (inputValue >= Float.parseFloat(ruleValue) && current.getRightChild() != null) {
                    current = current.getRightChild();
                } else {
                    // Broken tree, a decision node has null at one of its child nodes
                    throw new IllegalStateException("Broken tree: a decision node has null at one of its child nodes.");
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
            throw new IllegalStateException("No predictions could be made for this row.");
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
        // 1. Grab all column names (features and target) and extract target label
        String[] colNames = df.names();
        String targetLabel = colNames[colNames.length - 1];

        // Remove the target label to create an array of just features
        String[] features = new String[colNames.length - 1];
        System.arraycopy(colNames, 0, features, 0, colNames.length - 1);

        // 2. Keep track of the best label, its threshold, and its SSR (sum of squared residuals)
        String globalBestLabel = "";
        float globalBestThreshold = Float.POSITIVE_INFINITY;
        float globalBestSSR = Float.POSITIVE_INFINITY; // SSR = Sum of Squared Residuals

        // 3. Loop over every label (excluding target label)
            // a. Keep track of this labels best threshold and its SSR
            // b. Sort the values of the DataFrame according to the current label
            // c. Loop over all rows of the DataFrame
                // i. Grab current and next row
                // ii. Calculate current threshold (average of both rows for current label)
                // iii. Filter the dataset, grabbing all rows whose current label is less than current threshold
                // iv. Grab the average target label value from those rows
                // v. append a new column to iii., calculating the squared residual of each row
                // vi. repeate steps iii-v for rows whose current label is greater than or equal to the current threshold
                // vii. sum left and right squared residuals from the threshold into a combined sum
                // vii. update label best threshold if it has a SSR lower than the current best threshold
            // d. update global best label if this current label has a lower SSR value
        for (String label : features) {
            float labelBestThreshold = Float.POSITIVE_INFINITY;
            float labelBestSSR = Float.POSITIVE_INFINITY; 

            // Sort the values of the DataFrame for the current label
            DataFrame sortedDf = df.stream()
                .sorted((row1, row2) -> Float.compare(row1.getFloat(label), row2.getFloat(label)))
                .collect(Collectors.toDataFrame(df.schema()));

            // Loop over all rows, calculating a threshold for every 2 datapoints
            for (int i = 0; i < sortedDf.nrow() - 1; i++) {
                Tuple currentRow = sortedDf.apply(i);
                Tuple nextRow = sortedDf.apply(i+1);
                float currentThreshold = (currentRow.getFloat(label) + nextRow.getFloat(label)) / 2;

                DataFrame lessThanDf;
                DataFrame greaterThanDf;

                // Edge case: If no rows are returned by the filter, an error is raised
                try {
                    // Grabbing all rows that fall to the left of the threshold for the current label
                    lessThanDf = sortedDf.stream()
                        .filter(row -> row.getFloat(label) < currentThreshold)
                        .collect(Collectors.toDataFrame(df.schema()));
                } catch (IllegalArgumentException e) {
                    // No matches in the filter (rows = 0). No SSR can be calculated so continue to the next row
                    continue;
                }
                try {
                    // Grabbing all rows that fall at or to the right of the threshold for the current label
                    greaterThanDf = sortedDf.stream()
                        .filter(row -> row.getFloat(label) >= currentThreshold)
                        .collect(Collectors.toDataFrame(sortedDf.schema()));
                } catch (IllegalArgumentException e) {
                    // No matches in the filter (rows = 0). No SSR can be calculated so continue to the next row
                    continue;
                }

                // Calculating average target label value for datapoints to the left of the threshold
                ValueVector leftVals = lessThanDf.apply(targetLabel);
                float leftAverage;
                float leftTotal = 0.0f;
                for (int j = 0; j < leftVals.size(); j++) {
                    leftTotal +=  leftVals.getFloat(j);
                }
                leftAverage = leftTotal / leftVals.size();

                // Calculating average target label value for datapoints to the right of the threshold
                ValueVector rightVals = greaterThanDf.apply(targetLabel);
                float rightAverage;
                float rightTotal = 0.0f;
                for (int j = 0; j < rightVals.size(); j++) {
                    rightTotal +=  rightVals.getFloat(j);
                }
                rightAverage = rightTotal / rightVals.size();

                // Calculate the squared residual for each datapoint
                List<Float> leftResiduals = new ArrayList<>();
                for (int j = 0; j < leftVals.size(); j++) {
                    float residual = (float) Math.pow(leftVals.getDouble(j) - leftAverage, 2);
                    leftResiduals.add(residual);
                }
                List<Float> rightResiduals = new ArrayList<>();
                for (int j = 0; j < rightVals.size(); j++) {
                    float residual = (float) Math.pow(rightVals.getDouble(j) - rightAverage, 2);
                    rightResiduals.add(residual);
                }

                // Calculate the combined SSR by adding the left and right SSR's
                float leftSSR = (float) leftResiduals.stream().mapToDouble(Float::doubleValue).sum();
                float rightSSR = (float) rightResiduals.stream().mapToDouble(Float::doubleValue).sum();
                float combinedSSR = leftSSR + rightSSR;

                // If this SSR is < current labels best, update labels best
                if (combinedSSR < labelBestSSR) {
                    labelBestSSR = combinedSSR;
                    labelBestThreshold = currentThreshold;
                }
            }

            // If this labels best SSR (minimized) < current cadidate best SSR, update labels
            if (labelBestSSR <  globalBestSSR) {
                globalBestSSR = labelBestSSR;
                globalBestThreshold = labelBestThreshold;
                globalBestLabel = label;
            }
        }

        // 4. Now, the best candidate is chose for the root node. 
        // Create the root node, passing in a string containing the global best label and its threshold to the rule
        TreeNode root = new TreeNode(String.format("%s < %f", globalBestLabel, globalBestThreshold));

        // 5. Filter the dataframe into rows less than and rows greater than or equal to the global threshold
        final String splitLabel;
        final float splitThreshold;

        // Edge case: If global best label is "" and threshold is INFINITY, each feature label had the same values for its rows
            // Ex: correctLast3 = 0 for all rows in df, avgTime = 32.9 for all rows in df, etc.
            // Causes the continue condition to execute, unable to calculate any SSR's, leaving the globals as default
            // If so, randomly pick a label and choose the first row as threshold
                // Row for threshold doesn't matter since they are all the same values
        if (globalBestLabel == "" && globalBestThreshold == Float.POSITIVE_INFINITY) {
            int randomLabelIndex = (int) Math.random() * features.length;
            splitLabel = features[randomLabelIndex];
            splitThreshold = df.apply(splitLabel).getFloat(0);
        } else {
            splitLabel = globalBestLabel;
            splitThreshold = globalBestThreshold;
        }

        // Collect df's into lists to prevent the need for a try catch block
        List<Row> lessThanRows = df.stream()
            .filter(row -> row.getFloat(splitLabel) < splitThreshold)
            .collect(java.util.stream.Collectors.toList());
        List<Row> greaterThanRows = df.stream()
            .filter(row -> row.getFloat(splitLabel) >= splitThreshold)
            .collect(java.util.stream.Collectors.toList());

        // 6. Conditionally check to crete a leaf node or continue splitting based on the rows returned by the split, minSamples, and maxDepth
        if (lessThanRows.size() == 0) {
            return null;
        }
        else {
            // Some rows are returned, so create the DataFrame
            DataFrame lessThanDf = df.stream()
                .filter(row -> row.getFloat(splitLabel) < splitThreshold)
                .collect(Collectors.toDataFrame(df.schema()));
            ValueVector lessThanVector = lessThanDf.apply(targetLabel);

            if (lessThanRows.size() < minSamples || currentDepth >= maxDepth) {
                // Average prediction
                float averagePrediction = 0.0f;
                for (Double val : lessThanVector.toDoubleArray()) {
                    averagePrediction += val.floatValue();
                }
                averagePrediction = lessThanVector.size() == 0 ? 0 : averagePrediction / lessThanVector.size();

                // Create a left leaf node
                TreeNode leftLeafNode = new TreeNode(averagePrediction);
                root.setLeftChild(leftLeafNode);
            } else {
                // Continue splitting to create the left sub-tree
                root.setLeftChild(createRegressionTree(lessThanDf, minSamples, maxDepth, currentDepth + 1));
            }
        }
        if (greaterThanRows.size() == 0) {
            return null;
        }
        else {
            // Some rows are returned, so create the DataFrame
            DataFrame greaterThanDf = df.stream()
                .filter(row -> row.getFloat(splitLabel) >= splitThreshold)
                .collect(Collectors.toDataFrame(df.schema()));
            ValueVector greaterThanVector = greaterThanDf.apply(targetLabel);

            if (greaterThanRows.size() < minSamples || currentDepth >= maxDepth) {
                // Average prediction
                float averagePrediction = 0.0f;
                for (Double val : greaterThanVector.toDoubleArray()) {
                    averagePrediction += val.floatValue();
                }
                averagePrediction = greaterThanVector.size() == 0 ? 0 : averagePrediction / greaterThanVector.size();

                // Create a right leaf node
                TreeNode rightLeafNode = new TreeNode(averagePrediction);
                root.setRightChild(rightLeafNode);
            } else {
                // Continue splitting to create the right sub-tree
                root.setRightChild(createRegressionTree(greaterThanDf, minSamples, maxDepth, currentDepth + 1));
            }
        }

        // Return the root node
        return root;
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
        DataFrame bootstrappedDf = df.apply(index);

        // 5. Return that dataframe
        return bootstrappedDf;
    }
}
