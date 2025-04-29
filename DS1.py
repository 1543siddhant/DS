# 1 Data Wrangling, I
# Perform the following operations using Python on any open source dataset (e.g., data.csv)
# 1. Import all the required Python Libraries.
# 2. Locate open source data from the web (e.g., https://www.kaggle.com). Provide a clear description of the data and its source (i.e., URL of the web site).
# 3. Load the Dataset into pandas dataframe.
# 4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() function to get some initial statistics. Provide variable descriptions. Types of variables etc. Check the dimensions of the data frame.
# 5. Data Formatting and Data Normalization: Summarize the types of variables by checking the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set. If variables are not in the correct data type, apply proper type conversions.
# 6. Turn categorical variables into quantitative variables in Python.
# In addition to the codes and outputs, explain every operation that you do in the above steps and explain everything that you do to import/read/scrape the data set.

# Step 1: Import all the required Python Libraries
import pandas as pd
import numpy as np

# Step 3: Load the Dataset into pandas DataFrame
df = pd.read_csv('train.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Step 4: Data Preprocessing
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Initial statistics
print("\nSummary statistics:")
print(df.describe())

# Variable descriptions
print("\nVariable Descriptions:")
print("""
- PassengerId: Unique ID for each passenger (integer)
- Survived: Survival status (0 = No, 1 = Yes) (integer, categorical)
- Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd) (integer, ordinal)
- Name: Passenger name (string)
- Sex: Gender (male, female) (string, categorical)
- Age: Age in years (float, numeric)
- SibSp: Number of siblings/spouses aboard (integer, numeric)
- Parch: Number of parents/children aboard (integer, numeric)
- Ticket: Ticket number (string)
- Fare: Passenger fare (float, numeric)
- Cabin: Cabin number (string, missing for many)
- Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) (string, categorical)
""")

# Check dimensions
print("\nDimensions of the DataFrame:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Step 5: Data Formatting and Data Normalization
# Summarize variable types
print("\nData types of each column:")
print(df.dtypes)

# Type conversions (if needed)
# Convert 'Survived' and 'Pclass' to categorical (since they are ordinal/categorical)
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')

# Verify conversions
print("\nData types after conversion:")
print(df.dtypes)

# Step 6: Turn categorical variables into quantitative variables
# Identify categorical variables: 'Sex', 'Embarked'
# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
print("\nFirst 5 rows after one-hot encoding:")
print(df_encoded.head())

# Display new columns
print("\nColumns after encoding:")
print(df_encoded.columns)

