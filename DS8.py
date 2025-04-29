# 8) Data Visualization I
# 1. Use the inbuilt dataset 'titanic'. The dataset contains 891 rows and contains information about the passengers who boarded the unfortunate Titanic ship. Use the Seaborn library to see if we can find any patterns in the data.
# 2. Write a code to check how the price of the ticket (column name: 'fare') for each passenger
# is distributed by plotting a histogram.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Load the Titanic dataset
titanic = sns.load_dataset('titanic')
# Step 1: Explore patterns in the data with Seaborn visualizations
print("=== Exploring Patterns in Titanic Dataset ===")
# Set the style for better visuals
sns.set_style("whitegrid")
# Plot 1: Survival rate by passenger class
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='pclass', hue='survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
# Plot 2: Survival rate by sex
plt.figure(figsize=(8, 5))
sns.countplot(data=titanic, x='sex', hue='survived')
plt.title('Survival Count by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
# Plot 3: Age distribution by survival
plt.figure(figsize=(8, 5))
sns.histplot(data=titanic, x='age', hue='survived', multiple='stack', bins=30)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()
# Plot 4: Fare vs Age with survival
plt.figure(figsize=(8, 5))
sns.scatterplot(data=titanic, x='age', y='fare', hue='survived', style='pclass', size='pclass')
plt.title('Fare vs Age by Survival and Class')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['No', 'Yes'], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
# Step 2: Histogram of ticket prices (fare)
print("\n=== Distribution of Ticket Prices (Fare) ===")
plt.figure(figsize=(8, 5))
sns.histplot(data=titanic, x='fare', bins=30, kde=True)
plt.title('Distribution of Ticket Prices (Fare)')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()
# Basic statistics for fare
print("\nFare Statistics:")
print(titanic['fare'].describe())