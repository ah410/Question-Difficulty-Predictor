# Student Question Difficulty Readiness Prediction

This project aims to predict a student's readiness to be given harder questions based on historical data using a **Random Forest Regression Model**. The model is trained on various student statistics and predicts a readiness score for future questions, incorporating a decay function.

## Project Structure

- `model.py`: Contains the RandomForestRegressor model, with functions for training, predicting, and calculating a score based on inputs.
- `data.py`: Provides a function to load and preprocess the student data from a CSV file into a pandas DataFrame.
- `train_and_run.py`: The script used to train the model, perform predictions, and generate readiness scores based on student data.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/ah410/question_difficulty_predictor.git
   cd question_difficulty_predictor
