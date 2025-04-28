package com.example;
import smile.data.DataFrame;
import smile.io.Read;

public class Main {
    public static void main(String[] args) {
        try {
            DataFrame df = Read.csv("src/main/resources/dataset/student_statistics.csv", "header=true");
            RandomForestRegressor model = new RandomForestRegressor(5, 5, 10);
            model.bootstrapSample(df);

        } catch (java.io.IOException | java.net.URISyntaxException e) {
            System.err.println("Failed to read CSV file: " + e.getMessage());
        }
    }
}
