# 3) Descriptive Statistics - Measures of Central Tendency and variability

# Perform the following operations on any open source dataset (e.g., data.csv)
# 1. Provide summary statistics (mean, median, minimum, maximum, standard deviation) for a
# Curriculum for Third Year of Artificial Intelligence and Data Science (2019 Course), Savitribai Phule Pune University
# http://collegecirculars.unipune.ac.in/sites/documents/Syllabus2022/Forms/AllItems.aspx
# #84/105
# dataset (age, income etc.) with numeric variables grouped by one of the qualitative (categorical) variable. For example, if your categorical variable is age groups and quantitative variable is income, then provide summary statistics of income grouped by the age groups. Create a list that contains a numeric value for each response to the categorical variable.
# 2. Write a Python program to display some basic statistical details like percentile, mean, standard deviation etc. of the species of ‘Iris-setosa’, ‘Iris-versicolor’ and ‘Iris-versicolor’ of iris.csv dataset.
# Provide the codes with outputs and explain everything that you do in this step.

# Import required libraries
import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('train.csv')
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Group by categorical variable (Pclass) and compute statistics for Fare
grouped_stats = df.groupby('Pclass')['Fare'].agg(['mean', 'median', 'min', 'max', 'std']).round(2)
print("\nSummary statistics for Fare grouped by Pclass:")
print(grouped_stats)

# Step 3: Create a numeric list for the categorical variable
# Pclass is already numeric (1, 2, 3), but formalize as a list
pclass_numeric = df['Pclass'].unique().tolist()
pclass_numeric.sort()  # Ensure sorted order
print("\nNumeric values for Pclass categories:")
print(pclass_numeric)

# Part 2: Iris Dataset - Statistical Details by Species
print("\n=== Part 2: Iris Dataset ===")

# Load dataset
iris_df = pd.read_csv('Iris.csv')
print("\nFirst 5 rows of Iris dataset:")
print(iris_df.head())

# Drop Id column
iris_df = iris_df.drop('Id', axis=1)
print("\nDropped 'Id' column")

# Rename columns for consistency
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
print("\nRenamed columns to: sepal_length, sepal_width, petal_length, petal_width, species")

# Verify species values
print("\nSpecies values:")
print(iris_df['species'].unique())

# Select numeric columns
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# Check for missing values
print("\nMissing values in Iris dataset:")
print(iris_df.isnull().sum())

# Compute statistics by species
stats_by_species = iris_df.groupby('species')[numeric_cols].describe().round(2)

print("\nStatistical details by species:")
print(stats_by_species)

# Specific statistics
mean_stats = iris_df.groupby('species')[numeric_cols].mean().round(2)
std_stats = iris_df.groupby('species')[numeric_cols].std().round(2)
percentiles = iris_df.groupby('species')[numeric_cols].quantile([0.25, 0.50, 0.75]).round(2)

# Step 2: Explore the data
print("\n=== Dataset Info ===")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nBasic statistics:")
print(df.describe().round(2))

