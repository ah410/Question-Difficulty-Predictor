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

    }
    public float predict(DataFrame df) {
        return 0.0f;
    }
    public float decay(int daysSinceLastLoggedIn) {
        return 0.0f;
    }
    public TreeNode createRegressionTree(DataFrame df, int minSamples, int maxDepth, int currentDepth) {
        return new TreeNode("");
    }
    public DataFrame bootstrapSample(DataFrame df) {
        return DataFrame.of(new int[][]{{0,0}, {0,0}});
    }
}
