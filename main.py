import pandas as pd
import os

# A function to list the files and directories
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]

# Import the csv files and load into a Dataframe
filepaths = list_full_paths('./UK Used Cars')
df = pd.concat(map(pd.read_csv, filepaths))

# Print the first 10 lines
print(df.head())

# Descibe the Dataset
print(df.info)
print(df.describe())

# Supervised Learning / Regression Model
# Checking for multicolinerality
# Scalar
# Score function
# knn
# decision tree
# Hyperprameter
# values count

# Clean the Data


