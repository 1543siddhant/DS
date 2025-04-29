# 2) Data Wrangling II

# Create an “Academic performance” dataset of students and perform the following operations using Python.

# 1. Scan all variables for missing values and inconsistencies. If there are missing values and/or inconsistencies, use any of the suitable techniques to deal with them.
# 2. Scan all numeric variables for outliers. If there are outliers, use any of the suitable techniques to deal with them.
# 3. Apply data transformations on at least one of the variables. The purpose of this transformation should be one of the following reasons: to change the scale for better understanding of the variable, to convert a non-linear relation into a linear one, or to decrease the skewness and convert the distribution into a normal distribution.
# Reason and document your approach properly.

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Step 1: Create the "Academic Performance" dataset
np.random.seed(42)  # For reproducibility
n_students = 100

data = {
    'Student_ID': [f'S{i:03d}' for i in range(1, n_students + 1)],
    'Name': [f'Student_{i}' for i in range(1, n_students + 1)],
    'Age': np.random.randint(15, 19, n_students),
    'Gender': np.random.choice(['Male', 'Female', np.nan], n_students, p=[0.48, 0.48, 0.04]),
    'Study_Hours': np.random.exponential(scale=10, size=n_students).round(1),
    'Attendance': np.random.uniform(50, 100, n_students).round(1),
    'Math_Score': np.random.normal(70, 15, n_students).round(1),
    'Science_Score': np.random.normal(65, 15, n_students).round(1),
    'GPA': np.random.uniform(2.0, 4.0, n_students).round(2)
}

# Introduce missing values and inconsistencies
df = pd.DataFrame(data)
df.loc[np.random.choice(df.index, 5), 'Study_Hours'] = np.nan  # 5 missing Study_Hours
df.loc[np.random.choice(df.index, 3), 'Math_Score'] = -10  # Inconsistent scores
df.loc[np.random.choice(df.index, 2), 'Age'] = 20  # Inconsistent ages
df.loc[np.random.choice(df.index, 3), 'Attendance'] = 110  # Inconsistent attendance

print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Scan for missing values and inconsistencies
# Missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Handle missing values (fixing the warning)
df['Study_Hours'] = df['Study_Hours'].fillna(df['Study_Hours'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Inconsistencies
print("\nInconsistencies check:")
print("Invalid Ages (not 15–18):", df[~df['Age'].between(15, 18)]['Age'].count())
print("Negative Math_Score:", df[df['Math_Score'] < 0]['Math_Score'].count())
print("Negative Science_Score:", df[df['Science_Score'] < 0]['Science_Score'].count())
print("Invalid Attendance (not 0–100):", df[~df['Attendance'].between(0, 100)]['Attendance'].count())
print("Invalid GPA (not 0–4.0):", df[~df['GPA'].between(0, 4.0)]['GPA'].count())

# Handle inconsistencies
df['Age'] = df['Age'].where(df['Age'].between(15, 18), df['Age'].mode()[0])
df['Math_Score'] = df['Math_Score'].where(df['Math_Score'] >= 0, df['Math_Score'].median())
df['Science_Score'] = df['Science_Score'].where(df['Science_Score'] >= 0, df['Science_Score'].median())
df['Attendance'] = df['Attendance'].where(df['Attendance'].between(0, 100), df['Attendance'].median())
df['GPA'] = df['GPA'].where(df['GPA'].between(0, 4.0), df['GPA'].median())

print("\nInconsistencies after correction:")
print("Invalid Ages (not 15–18):", df[~df['Age'].between(15, 18)]['Age'].count())
print("Negative Math_Score:", df[df['Math_Score'] < 0]['Math_Score'].count())
print("Negative Science_Score:", df[df['Science_Score'] < 0]['Science_Score'].count())
print("Invalid Attendance (not 0–100):", df[~df['Attendance'].between(0, 100)]['Attendance'].count())
print("Invalid GPA (not 0–4.0):", df[~df['GPA'].between(0, 4.0)]['GPA'].count())

# Step 3: Scan numeric variables for outliers
numeric_cols = ['Age', 'Study_Hours', 'Attendance', 'Math_Score', 'Science_Score', 'GPA']

# Function to detect and cap outliers
def cap_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

# Apply to numeric columns
for col in numeric_cols:
    print(f"\nOutliers in {col}:")
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
    print(f"Number of outliers: {len(outliers)}")
    df[col] = cap_outliers(df[col])


# Step 4: Apply data transformation
# Variable: Study_Hours (reduce skewness)
print("\nSkewness of Study_Hours before transformation:", skew(df['Study_Hours']))

# Log transformation
df['Log_Study_Hours'] = np.log1p(df['Study_Hours'])

print("Skewness of Log_Study_Hours after transformation:", skew(df['Log_Study_Hours']))

# Visualize transformation
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Study_Hours'], bins=20, edgecolor='k')
plt.title('崀Study_Hours Distribution')
plt.subplot(1, 2, 2)
plt.hist(df['Log_Study_Hours'], bins=20, edgecolor='k')
plt.title('Log_Study_Hours Distribution')
plt.tight_layout()
plt.show()

print("\nFirst 5 rows after all operations:")
print(df.head())

