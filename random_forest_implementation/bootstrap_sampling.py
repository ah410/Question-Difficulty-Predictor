def bootstrap_sampling(dataset):
    # 1. Grab number of rows of original dataset
    # 2. Create an empty DataFrame to store the bootstrap sample
    # 3. For size N of original dataset
        # a. Generate a random index in range [0,N-1] inclusive
        # b. add the row of that index to the DataFrame
    # 4. Now, you have a new DataFrame of size N, with replacement
    # 5. Return that dataframe
    ...