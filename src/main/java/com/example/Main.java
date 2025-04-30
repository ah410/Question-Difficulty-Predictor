package com.example;

import java.io.File;

import smile.data.DataFrame;
import smile.data.Row;
import smile.io.Read;

public class Main {
    public static void main(String[] args) {
        // 1. Generate dummy test data
        float[][] testData = {
            {389.15f, 2f, 69.41f, 0.77f, 0.64f, 19f, 0.6891f, 7f},
            {205.78f, 1f, 45.32f, 0.53f, 0.41f, 14f, 0.5127f, 4f},
            {312.99f, 3f, 58.23f, 0.91f, 0.79f, 22f, 0.7632f, 9f},
            {97.46f, 0f, 82.51f, 0.28f, 0.15f, 5f, 0.3129f, 2f},
            {654.33f, 3f, 60.04f, 0.96f, 0.92f, 30f, 0.8745f, 15f},
            {148.27f, 2f, 67.12f, 0.65f, 0.38f, 10f, 0.4873f, 3f},
            {219.88f, 1f, 49.78f, 0.44f, 0.53f, 12f, 0.5216f, 6f},
            {443.12f, 3f, 72.45f, 0.88f, 0.99f, 24f, 0.8159f, 11f},
            {330.67f, 2f, 64.89f, 0.79f, 0.68f, 16f, 0.6735f, 8f},
            {571.01f, 3f, 59.76f, 0.92f, 0.87f, 27f, 0.8341f, 13f}
        };
        DataFrame testDf = DataFrame.of(testData, "timeSpent","correctLast3","avgTime","confidence","quizScore","sessions","readinessScore","numberQuestionsAnswered");

        System.out.println("Test Data:");
        System.out.println(testDf.head(10));

        // 2. Create a variable to store the model on load or on creation
        RandomForestRegressor model;
        ModelSaveAndLoad saveAndLoad = new ModelSaveAndLoad();

        // 3. If the model is in a file, use it. Else, create and train the model
        File f = new File("RandomForestRegressor.model");
        if (f.exists()) {
            // Load the model
            model = saveAndLoad.loadModel("RandomForestRegressor.model");

            // Check if successful load
            if (model == null) {
                System.out.println("null model");
                return;
            } else {
                System.out.println("Number of trees in the forest: " + model.getForest().size());
                System.out.println("successful load of model");
                System.out.println("Testing prediction.");
            }
        }
        else {
            // Convert the training dataset (CSV file) to a Smile DataFrame
            DataFrame df;
            try {
                df = Read.csv("src/main/resources/dataset/student_statistics.csv", "header=true");
            } 
            catch (java.io.IOException | java.net.URISyntaxException e) {
                System.err.println("Failed to read CSV file: " + e.getMessage());
                return;
            }
            
            // Create your model object from class RandomForestRegressor
            model = new RandomForestRegressor(100, 5, 10);

            // Train the model on the dataset
            System.out.println("Training the model...");
            model.fit(df); // Create nEstimator regression trees on this dataframe
            System.out.println("Model finished training...");

            // Save the model
            saveAndLoad = new ModelSaveAndLoad();
            saveAndLoad.saveModel(model, "RandomForestRegressor.model");
        }

        // 4. Test the model by calling predict()
        System.out.println("Calculating predictions...");
        for (Row row : testDf) {
            System.out.println(row.toString());
            float prediction = model.predict(row);
            System.out.println("SCORE: " + prediction);

        }
    }
}
