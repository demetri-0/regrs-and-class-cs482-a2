import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("real_estate_valuation_data.csv")

""" ******** Meet the Data ******** """

print(f"Number of features: {data.shape[1] - 1}\n")
print(f"Names of features:\n{list(data.columns[:-1])}\n")
print(f"Name of target: {data.columns[-1]}\n")
num_samples = data.shape[0]
print(f"Number of samples: {num_samples}\n")
print(f"First 5 rows:\n{data.head()}\n")

""" ******************************* """

""" ******** Understand the Data ******** """

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.25, 
                                                    shuffle=True, 
                                                    random_state=42)
