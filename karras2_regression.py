# CS 482 - Assignment 2
# Author: Demetri Karras
# File: karras2_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

""" ------ REGRESSION ------ """

data1 = pd.read_csv("real_estate_valuation_data.csv")

""" ******** Meet the Data ******** """

num_features = data1.shape[1] - 1
print(f"Number of Features: {num_features}\n")
print(f"Names of Features:\n{list(data1.columns[:-1])}\n")
print(f"Name of Target: {data1.columns[-1]}\n")
num_samples = data1.shape[0]
print(f"Number of Samples: {num_samples}\n")
print(f"First 5 Rows:\n{data1.head()}\n")

""" ******** Understand the Data ******** """

corr_matrix = data1.corr()  # create correlation matrix
print("Correlation Matrix")
print(f"{corr_matrix}\n")
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
print("\n")

model = LinearRegression()
model.fit(x_train, y_train)

# obtain metrics and display to user
training_r2 = model.score(x_train, y_train)
test_r2 = model.score(x_test, y_test)

y_pred = model.predict(x_test)
rmse = root_mean_squared_error(y_test, y_pred)

print("Linear Regression Metrics (pre cross validation)")
print(f"Training R^2: {training_r2}")
print(f"Test R^2: {test_r2}")
print(f"RMSE: {rmse}")

""" ******** Cross Validation ******** """
print("\n")

# 5-fold cross validation using linear regression
scores = cross_validate(LinearRegression(), x, y, cv=5, scoring=("r2", "neg_root_mean_squared_error"), return_train_score=True)
train_cv_r2_scores = np.append(scores["train_r2"], np.average(scores["train_r2"]))
test_cv_r2_scores = np.append(scores["test_r2"], np.average(scores["test_r2"]))
train_cv_rmse_scores = np.append(-scores["train_neg_root_mean_squared_error"], np.average(-scores["train_neg_root_mean_squared_error"]))
test_cv_rmse_scores = np.append(-scores["test_neg_root_mean_squared_error"], np.average(-scores["test_neg_root_mean_squared_error"]))

# consolidate score data into pd.DataFrame and display to user
header = [1, 2, 3, 4, 5, "Mean"]
display_data = {
    "Folds": header,
    "Training R^2 Scores": train_cv_r2_scores,
    "Test R^2 Scores": test_cv_r2_scores,
    "Training RMSE Scores": train_cv_rmse_scores,
    "Test RMSE Scores": test_cv_rmse_scores
}

cross_val_results_table = pd.DataFrame(display_data)
print("\nCross Validation Results")
print(cross_val_results_table.to_string(index=False))

""" ******** Model Evaluation and Analysis ******** """
print("\n")

epsilon_values = [0.2, 0.5, 1] # epsilon values to expiriment with
c_values = [1, 10, 100] # c values to experiment with

for e in epsilon_values:
    for c in c_values: # iterate through all combinations of epsilon and C

        # create pipeline to scale data and apply SVM regression using epsilon and C values
        svr_model = make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf", C=c, epsilon=e) 
        )
        svr_model.fit(x_train, y_train)    

        # consolidate metrics and display to user for comparison
        scores = cross_validate(svr_model, x, y, cv=5, scoring=("r2", "neg_root_mean_squared_error"), return_train_score=True)
        mean_train_cv_r2 = np.average(scores["train_r2"])
        mean_test_cv_r2 = np.average(scores["test_r2"])
        mean_train_cv_rmse = np.average(-scores["train_neg_root_mean_squared_error"])
        mean_test_cv_rmse = np.average(-scores["test_neg_root_mean_squared_error"])

        print(f"Mean Cross Validation Scores for epsilon = {e} and C = {c}")
        print(f"Training R^2: {mean_train_cv_r2}")
        print(f"Test R^2: {mean_test_cv_r2}")
        print(f"Training RMSE: {mean_train_cv_rmse}")
        print(f"Test RMSE: {mean_test_cv_rmse}")
        print("\n")
