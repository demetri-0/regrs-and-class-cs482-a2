import pandas as pd
from sklearn

""" ------ CLASSIFICATION ------ """

data2 = pd.read_csv("haberman.csv")

""" ******** Meet the Data ******** """

num_features = data2.shape[1] - 1
print(f"Number of Features: {num_features}\n")
print(f"Names of Features:\n{list(data2.columns[:-1])}\n")
print(f"Name of Target: {data2.columns[-1]}\n")
num_samples = data2.shape[0]
print(f"Number of Samples: {num_samples}\n")
print(f"First 5 Rows:\n{data2.head()}\n")

""" ******** Understand the Data ******** """

corr_matrix = data2.corr()  # create correlation matrix
print("Correlation Matrix")
print(f"{corr_matrix}\n")
target_name = data2.columns[-1]
features_to_drop = set()

corr_matrix_columns = corr_matrix.columns
for i in range(0, len(corr_matrix_columns)):
    for j in range(i + 1, len(corr_matrix_columns)): # iterate through each cell of the upper triangular section of the matrix 

        row_var = corr_matrix_columns[i]
        col_var = corr_matrix_columns[j]
        value = corr_matrix.iloc[i, j]

        # check feature-to-feature correlation
        if (row_var != target_name and col_var != target_name) and (value >= 0.9 or value <= -0.9):
            features_to_drop.add(row_var)

        # check feature-to-target weak correlation
        if (row_var == target_name or col_var == target_name) and (-0.05 <= value <= 0.05):
            non_target = row_var if col_var == target_name else col_var # ensure target is not marked for removal
            features_to_drop.add(non_target)

# drops features that were added above
for feature in features_to_drop:
    data2.drop(feature, axis=1, inplace=True)
    print(f"{feature} has been dropped due to correlation.")

num_features = data2.shape[1] - 1  # recalculate in case any features were dropped
