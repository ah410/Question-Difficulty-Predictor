"""
validation.py: Incorporates testing and validation into the training of the RandomForestRegressor model.

Includes:
    k_fold_cross validation function (trains the model and evaluates its accuracy using 5 folds)
        - outputs a validation scores, average score, and a plot of the accuracy of the 5 folds
    grid_search function (called first to find the best parameters for the function)
        - uses parameters to exhaustively search for the best ones

Notes:
    I manually ran grid_search() first then used the output to statically put in the parameters
    in k_fold_cross_validation() for the RandomForestRegressor() model. However, if you would like to use
    this file, you will likely want to call grid_search() first in main, then grab the outputs of
    grid_search to pass into k_fold_cross_validation(). I will update this file in the future to reflect
    these thoughts.
"""

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def k_fold_cross_validation(student_dataset):
    X = student_dataset.drop(columns=['timeSpent', 'readinessScore', 'numberQuestionsAnswered'])
    Y = student_dataset['readinessScore']

    # Grid search best parameters found: n_estimators=1000, max_depth=50, min_samples_leaf=1 (DEFAULT), min_samples_split=2 (DEFAULT)
    model = RandomForestRegressor(n_estimators=1000, max_depth=50)
    scores = cross_val_score(model, X, Y, cv=5)

    print(f'cross validation scores: {scores}', end='')
    print(f' | average score: {sum(scores)/len(scores)}\n')

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', color='forestgreen')
    plt.xticks(range(1, len(scores) + 1))
    plt.ylim((0,1))
    plt.xlabel('Fold')
    plt.ylabel('R^2 Score (Higher is Better)')
    plt.title('Random Forest Cross-Validation R^2 Scores')
    plt.grid(True)

    # TODO: Add some logic that grabs the latest revision number, and increments so no images are overwritten
    plt.savefig('./dataset/model_accuracy/model_revision_3.jpg')

    for x,y in zip(range(1, len(scores) + 1), scores):
        plt.text(x, y, f'({x}, {round(y, 4)})', fontweight=700)

    plt.show()


def grid_search(student_dataset):
    X = student_dataset.drop(columns=['timeSpent', 'readinessScore', 'numberQuestionsAnswered'])
    Y = student_dataset['readinessScore']

    model = RandomForestRegressor()

    # Different model parameters to test
    parameters = {
        'n_estimators': [50, 100, 200, 500, 1000],
        'max_depth': [None, 5, 10, 50],
        'min_samples_split': [2, 5, 9],
        'min_samples_leaf': [1, 3, 6]
    }

    grid_search = GridSearchCV(model, parameters, cv=5, scoring='r2', verbose=2)
    grid_search.fit(X, Y)

    # Print best parameters and score
    print("Best Parameters:", grid_search.best_params_)
    print("Best R² Score:", grid_search.best_score_)
