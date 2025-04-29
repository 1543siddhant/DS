import pandas as pd
import matplotlib.pyplot as plt
# Download the Iris dataset from the provided URL
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris_df = pd.read_csv(url, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
iris_df.head()
# List down the features and their types
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feature_types = ['numeric', 'numeric', 'numeric', 'numeric']
# Print the features and their types
for feature, ftype in zip(features, feature_types):
    print(f"Feature: {feature} - Type: {ftype}")
# Create histograms for each feature
iris_df[features].hist()
plt.suptitle('Histograms of Iris Dataset Features')
plt.tight_layout()
plt.show()
# Create box plots for each feature
iris_df[features].plot(kind='box')
plt.title('Box Plots of Iris Dataset Features')
plt.show()

# Identify outliers
outliers = []
for feature in features:
    q1 = iris_df[feature].quantile(0.25)
    q3 = iris_df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    feature_outliers = iris_df[(iris_df[feature] < lower_bound) | (iris_df[feature] > upper_bound)]
    outliers.append(feature_outliers)

# Print the outliers for each feature
for feature, outlier_df in zip(features, outliers):
    print(f"\nOutliers for feature: {feature}")
    print(outlier_df)

