"""
train_and_predict.py: Trains the model and predicts readiness score.

Steps:
    1. Load and split training data into X (inputs) and Y (output).
    2. Train the model using model.fit().
    3. Predict readiness score using model.predict_score().
"""

from model import DifficultyModel

def train_and_predict(student_dataset, implementation_type):
    # Create and train the RandomForestRegression model
    difficulty_model = DifficultyModel(implementation_type)
    difficulty_model.train(student_dataset)

    # Grab some sample data to test the prediction of the model
    columns = ['correctLast3', 'avgTime', 'confidence', 'quizScore', 'sessions', 'lastLoggedIn']
    test_stats = [
        [2, 71.91, 0.83, 0.29, 17, 3],
        [3, 67.92, 0.76, 0.80, 28, 5],
        [1, 22.56, 0.57, 1.00, 9, 1],
        [3, 52.34, 0.90, 0.65, 15, 2],
        [0, 80.12, 0.85, 0.45, 30, 6],
        [2, 64.50, 0.78, 0.55, 22, 4],
        [3, 40.21, 0.68, 0.60, 18, 10],
        [2, 58.74, 0.75, 0.70, 25, 1],
        [0, 35.14, 0.72, 0.50, 12, 14],
        [1, 91.47, 0.88, 0.80, 20, 3],
        [3, 35.00, 0.99, 1.00, 25, 2]
    ]

    # Run the predictions
    difficulty_model.predict_score(columns, test_stats)
