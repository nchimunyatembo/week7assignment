import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data loading and exploration steps
# Load the Iris dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Explore the structure of the dataset
print("\nDataset info:")
print(data.info())
print("\nMissing values:")
print(data.isnull().sum())

# Clean the dataset (no missing values in this dataset, but this is how you would handle it)
data = data.dropna()

# Basic Data Analysis
print("\nBasic statistics:")
print(data.describe())

# Grouping and computing the mean for a numerical column per categorical group
species_mean = data.groupby('species').mean()
print("\nMean values for each species:")
print(species_mean)

# Visualizations
plt.figure(figsize=(12, 10))

# 1. Line chart (example: plotting sepal length trend across the dataset index)
plt.subplot(2, 2, 1)
plt.plot(data.index, data['sepal_length'], label='Sepal Length', color='blue')
plt.title("Sepal Length Trend")
plt.xlabel("Index")
plt.ylabel("Sepal Length")
plt.legend()

# 2. Bar chart (average petal length per species)
plt.subplot(2, 2, 2)
sns.barplot(x=species_mean.index, y=species_mean['petal_length'], palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")

# 3. Histogram (distribution of sepal width)
plt.subplot(2, 2, 3)
plt.hist(data['sepal_width'], bins=10, color='orange', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width")
plt.ylabel("Frequency")

# 4. Scatter plot (sepal length vs petal length)
plt.subplot(2, 2, 4)
sns.scatterplot(x=data['sepal_length'], y=data['petal_length'], hue=data['species'], palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")

plt.tight_layout()
plt.show()

# Findings or Observations
# 1. Sepal length shows a steady trend with varying species.
# 2. Setosa has the smallest petal lengths on average, while virginica has the largest.
# 3. Sepal width is mostly distributed between 2.0 and 4.5 cm.
# 4. A positive correlation exists between sepal length and petal length, differing by species.
