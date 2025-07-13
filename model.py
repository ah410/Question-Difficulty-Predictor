"""
model.py: Define your model class inheriting from sklearn's RandomForestRegressor.

Includes:
    - constructor
    - train function
    - decay function (will be multiplied with the predicted score to obtain the final score)
    - predict_score function (with decay functionality)
    - get_difficulty function
"""

import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from randomforestregression.random_forest_regression import RandomForestRegressor2

class DifficultyModel():
    def __init__(self, implementation_type):
        if implementation_type == "sklearn_implementation":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=10)
        else:
            self.model = RandomForestRegressor2(n_estimators=5, min_samples=5, max_depth=10)

        self.training_done = False
        self.implementation_type = implementation_type

    def train(self, df):
        """
        Train the RandomForestRegressor model using the provided training data.

        Parameters
        ----------
        df : pandas.DataFrame 
            A DataFrame containing the features [correctLast3, avgTime, confidence, quizScore, sessions].
        """
        if self.implementation_type == "sklearn_implementation":
            X = df.drop(columns=['timeSpent', 'readinessScore', 'numberQuestionsAnswered'])
            Y = df['readinessScore']

            self.model.fit(X, Y)
        else:
            X = df.drop(columns=['timeSpent', 'numberQuestionsAnswered'])
            self.model.fit(X)

        self.training_done = True

    def decay(self, days_since_last_login):
        """
        A decay function.

        This function applies a decay to the output of the model.predict_score() function
        according to the number of days since the user last logged in. A decay score ranges between
        [0.10, 1.00]. A larger output closer to 1.00 implies no decay and a smaller output implies 
        greater decay.

        Parameters
        ----------
        days_since_last_login : int
            The number of days since a user has last logged in
        """
        try:
            if days_since_last_login < 0:
                raise ValueError('Value is not zero or a positive integer.')
            
            if days_since_last_login >= 30:
                return 0.10
            elif days_since_last_login >= 3:
                return 1 - math.log(days_since_last_login, 10) + 0.477
            else:
                return 1.00
        except ValueError as e:
            print(f'Invalid input: {e}')

    def predict_score(self, columns, test_stats):
        """
        A prediction function.

        This function calls both the decay and predict function from the model based on user 
        provided input to output a final score that signifies how ready a user is for the next 
        jump in question difficulty. The decay rate is multipled by the model's predicted 
        score from the input.

        Parameters
        ----------
        columns : list[str]
            In this order, 
            expects ['correctLast3', 'avgTime', 'confidence', 'quizScore', 'sessions', 'lastLoggedIn']

        test_stats : list[list[float]] 
            Each row in the 2-D Array contains 6 columns following the columns data values.
            The last colulmn contians the number of days since last login, which is expected to come 
            from some type of database that gets updated every time a user los into the application.
        """
        for row in test_stats:
            last_logged_in = row[-1]
            x_input = pd.DataFrame([row], columns=columns).drop(columns=['lastLoggedIn'])
            
            decay_rate = self.decay(last_logged_in)
            model_score = self.model.predict(x_input)

            print(f'decay rate: {decay_rate} | model score: {model_score}', end='')
            print(f'predicted score: {decay_rate * model_score} | input row: {row}\n')


    def get_difficulty(self, predicted_score):
        """
        A difficulty function.

        This function checks the predicted score and returns whether the question will be easy, medium,
        or hard for that specific classroom subject and grade level.

        Parameters
        ----------
        predicted_score : float
            A score of how prepared the user is to take on the next question.
        """
        try:
            if predicted_score < 0 or predicted_score > 1:
                raise ValueError("Predicted score is outside normalization range, [0,1].")
            
            if predicted_score < 0.30:
                return 'easy'
            elif predicted_score >= 0.30 and predicted_score < 0.65:
                return 'medium'
            else:
                return 'hard'
        except ValueError as e:
            print(f'Invalid predicted score: {e}')
