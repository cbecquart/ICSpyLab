from icspy import ICS
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Instantiate ICS object
# ics = ICS() # default parameters
ics = ICS()

# Fit and transform the ICS model (equivalent of the function ICS-S3() from the R package ICS)
ics.fit_transform(X)

# Printing a summary
ics.describe()

# # Load the Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Create a logistic regression model with ICS as a preprocessing step
# ics = ICS()
# model = LogisticRegression(max_iter=200)
#
# # Train the model on the training set
# ics.fit(X_train)
# X_train_ics = ics.transform(X_train)
# model.fit(X_train_ics, y_train)
#
# # Make predictions on the test set
# X_test_ics = ics.transform(X_test)
# y_pred = model.predict(X_test_ics)
# print(X_test_ics.shape)



