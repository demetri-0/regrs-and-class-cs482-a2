# CS 482 - Assignment 2
# Author: Demetri Karras
# File: karras2_classification.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt

""" ------ CLASSIFICATION ------ """

data2 = pd.read_csv("haberman.csv")

""" ******** Meet the Data ******** """

num_features = data2.shape[1] - 1
print(f"Number of Features: {num_features}\n")
print(f"Names of Features:\n{list(data2.columns[:-1])}\n")
print(f"Name of Target: {data2.columns[-1]}\n")
print(f"Number of Classes: {data2.iloc[:, -1].nunique()}\n")
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

""" ******** Model Fitting ******** """
print("\n")

# split data and display histograms for each feature
x = data2.iloc[:, :-1]
y = data2.iloc[:, -1]  # target column
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.20, 
                                                    shuffle=True, 
                                                    random_state=42)


logreg_model = LogisticRegression(penalty=None, solver='lbfgs')
logreg_model.fit(x_train, y_train)

y_pred = logreg_model.predict(x_test)

# create and display confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
classf_report = classification_report(y_test, y_pred)

print("Confusion Matrix")
print(conf_matrix)

print("\nClassification Report")
print(classf_report)

""" ******** Cross Validation ******** """
print("\n")

# apply 5-fold cross validation, displaying data to user
scores = cross_validate(LogisticRegression(penalty=None, solver='lbfgs'), x, y, return_train_score=True)

train_cv_accuracy = np.append(scores["train_score"], np.average(scores["train_score"]))
test_cv_accuracy = np.append(scores["test_score"], np.average(scores["test_score"]))

header = [1, 2, 3, 4, 5, "Mean"]
display_data = {
    "Folds": header,
    "Training Accuracies": train_cv_accuracy,
    "Test Accuracies": test_cv_accuracy
}

cross_val_results_table = pd.DataFrame(display_data)
print(f"Cross Validation Results\n{cross_val_results_table.to_string(index=False)}")

""" ******** Threshold Analysis ******** """
print("\n")

y_pred_probs = logreg_model.predict_proba(x_test)[:, 1] # gets probabilites for Surv_status=2 from x_test

# obtain FPR and TPR values to plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs, pos_label=2)

fig, axis = plt.subplots()

axis.set_title("ROC Curve")
axis.set_xlabel("False Positive Rate")
axis.set_ylabel("True Positive Rate")
plt.plot(fpr, tpr)

plt.show()

# calculate min distance to (1, 0) to get best threshold
distances_to_upper_left = np.sqrt((0 - fpr)**2 + (1 - tpr)**2)
min_distance_ind = np.argmin(distances_to_upper_left)
best_threshold = thresholds[min_distance_ind]
best_fpr = fpr[min_distance_ind]
best_tpr = tpr[min_distance_ind]

data_display = {
    "Threshold": thresholds,
    "FPR": fpr,
    "TPR": tpr
}
print(pd.DataFrame(data_display).to_string(index=False))

print(f"\nBest Threshold: {best_threshold}")
print(f"Best FPR: {best_fpr}")
print(f"Best TPR: {best_tpr}")
