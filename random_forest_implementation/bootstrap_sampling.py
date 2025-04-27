import pandas as pd
import random

def bootstrap_sampling(dataset):
    # 1. Grab number of rows of original dataset
    n = len(dataset)

    # 2. Make an empty list to store the random indices for all rows to be used in the bootstrapped dataset
    random_indices = []

    # 3. For size N of original dataset
        # a. Generate a random index in range [0,N-1] inclusive and add to random_indices list
    for _ in range(n):
        random_indices.append(random.randint(0, n - 1))

    # 4. Create a new DataFrame passing in the random_indices list
    bootstrapped_dataset = dataset.iloc[random_indices]

    # 5. Return that dataframe
    return bootstrapped_dataset

dataset = pd.read_csv('./dataset/dosage_effectiveness.csv')
bootstrapped_dataset = bootstrap_sampling(dataset)

print(dataset.head(5))
print(bootstrapped_dataset.head(5))
