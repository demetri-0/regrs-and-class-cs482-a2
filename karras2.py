import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

""" ------ REGRESSION ------ """

data1 = pd.read_csv("real_estate_valuation_data.csv")

""" ******** Meet the Data ******** """

num_features = data1.shape[1] - 1
print(f"Number of features: {num_features}\n")
print(f"Names of features:\n{list(data1.columns[:-1])}\n")
print(f"Name of target: {data1.columns[-1]}\n")
num_samples = data1.shape[0]
print(f"Number of samples: {num_samples}\n")
print(f"First 5 rows:\n{data1.head()}\n")

""" ******** Understand the Data ******** """

corr_matrix = data1.corr()  # create correlation matrix
target_name = data1.columns[-1]
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
    data1.drop(feature, axis=1, inplace=True)
    print(f"{feature} has been dropped due to correlation.")

num_features = data1.shape[1] - 1  # recalculate in case any features were dropped

# split data and display histograms for each feature
x = data1.iloc[:, :-1]
y = data1.iloc[:, -1]  # target column
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    shuffle=True, 
                                                    random_state=42)

fig1, axes1 = plt.subplots(2, 3)  # create a 2x3 plot to display 5 histograms

f_ind = 0  # index of feature
for i in range(0, 2):
    for j in range(0, 3):  # iterate through axes, creating histograms in the first 5 axes
        ax = axes1[i][j]

        if f_ind < num_features:  # creates histogram for feature
            feature = x_train.columns[f_ind]

            ax.hist(x_train[feature], bins=10)
            ax.set_title(f'Histogram for "{feature}"')
            ax.set_xlabel(feature)
            ax.set_ylabel("Frequency")

            f_ind += 1
        else:  # last empty axis is hidden
            ax.set_visible(False)

plt.tight_layout()
plt.show()

""" ******** Model Fitting ******** """

""" ******** Cross Validation ******** """

""" ------ CLASSIFICATION ------ """
