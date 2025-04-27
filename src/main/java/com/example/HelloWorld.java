package com.example;
import smile.data.DataFrame;
import smile.io.Read;

public class HelloWorld {
    public static void main(String[] args) {
        try {
            DataFrame df = Read.csv("src/main/resources/dataset/student_statistics.csv");
            System.out.println("First 5 rows of Data:");
            System.out.println(df.head(5));
        } catch (java.io.IOException | java.net.URISyntaxException e) {
            System.err.println("Failed to read CSV file: " + e.getMessage());
        }
    }
}
