

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
data = pd.read_csv("Social_Network_Ads.csv")
data.head()
data.tail()
# Separate the features (X) and the target variable (y)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X)
print(y)
# Perform label encoding on the 'Gender' column

"""
In machine learning projects, we usually deal with datasets having different categorical columns where some columns have their elements in the ordinal variable category for e.g a column income level having elements as low, medium, or high in this case we can replace these elements with 1,2,3. where 1 represents ‘low’  2  ‘medium’  and 3′ high’. Through this type of encoding, we try to preserve the meaning of the element where higher weights are assigned to the elements having higher priority.

Label Encoding :
Label Encoding is a technique that is used to convert categorical columns into numerical ones so that they can be fitted by machine learning models which only take numerical data. It is an important pre-processing step in a machine-learning project.
"""

le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
# Split the dataset into training and testing sets

"""
The train_test_split function of the sklearn.model_selection package in Python splits arrays or matrices into random subsets for train and test data, respectively.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an instance of the Logistic Regression model
logistic_regression = LogisticRegression()
# Train the model on the training data
logistic_regression.fit(X_train, y_train)
# Predict the labels for the test set
y_pred = logistic_regression.predict(X_test)
# Compute the confusion matrix

"""
A confusion matrix is a matrix that summarizes the performance of a machine learning model on a set of test data. It is often used to measure the performance of classification models, which aim to predict a categorical label for each input instance.
"""

confusion = confusion_matrix(y_test, y_pred)
# Extract the values from the confusion matrix

"""
True Positive (TP): It is the total counts having both predicted and actual values are Dog.
True Negative (TN): It is the total counts having both predicted and actual values are Not Dog.
False Positive (FP): It is the total counts having prediction is Dog while actually Not Dog.
False Negative (FN): It is the total counts having prediction is Not Dog while actually, it is Dog.
"""

TN = confusion[0, 0]  # True Negative
FP = confusion[0, 1]  # False Positive
FN = confusion[1, 0]  # False Negative
TP = confusion[1, 1]  # True Positive
# Compute the accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Compute the error rate
error_rate = (FP + FN) / (TP + TN + FP + FN)

# Compute the precision
precision = TP / (TP + FP)

# Compute the recall
recall = TP / (TP + FN)
# display the confusion matrix
print(confusion)
# display the accuracy
print(accuracy)
# display the error rate
print(error_rate)
# display the precision
print(precision)
# display the recall
print(recall)
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix:")
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
