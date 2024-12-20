import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
data_path = "path_to_telecom_dataset.csv"
df = pd.read_csv(data_path)

# Task 1: User Overview Analysis
# Step 1: Top 10 handsets
handset_counts = df['handset'].value_counts().head(10)
print("Top 10 handsets:")
print(handset_counts)

# Step 2: Top 3 handset manufacturers
manufacturer_counts = df['manufacturer'].value_counts().head(3)
print("Top 3 handset manufacturers:")
print(manufacturer_counts)

# Step 3: Top 5 handsets per top 3 manufacturers
top_3_manufacturers = manufacturer_counts.index
for manufacturer in top_3_manufacturers:
    top_handsets = df[df['manufacturer'] == manufacturer]['handset'].value_counts().head(5)
    print(f"Top 5 handsets for {manufacturer}:")
    print(top_handsets)

# Step 4: Recommendations
print("Marketing Recommendations:")
print("1. Focus marketing efforts on the most popular handsets and manufacturers.")
print("2. Target top handset manufacturers to partner for promotions.")
print("3. Identify unique features of popular handsets to highlight in campaigns.")

# Task 1.1: Aggregated User Behavior
aggregated_data = df.groupby('user_id').agg(
    xdr_sessions=('session_id', 'count'),
    total_duration=('session_duration', 'sum'),
    total_download=('download', 'sum'),
    total_upload=('upload', 'sum'),
    total_data_volume=('total_data_volume', 'sum')
).reset_index()

print("Aggregated User Behavior:")
print(aggregated_data.head())

# Task 1.2: Exploratory Data Analysis
# Handle missing values
def handle_missing_values(df):
    return df.fillna(df.mean())

aggregated_data = handle_missing_values(aggregated_data)

# Handle outliers using Z-score
from scipy.stats import zscore
z_scores = np.abs(zscore(aggregated_data.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).any(axis=1)
aggregated_data = aggregated_data[~outliers]

# Variable Description
print("Variable Descriptions:")
print(aggregated_data.info())

# Variable Transformation: Segment users into deciles
aggregated_data['decile'] = pd.qcut(aggregated_data['total_duration'], 10, labels=False)

decile_agg = aggregated_data.groupby('decile').agg(
    total_data=('total_data_volume', 'sum'),
    mean_duration=('total_duration', 'mean')
).reset_index()

print("Decile Analysis:")
print(decile_agg)

# Univariate Analysis (Non-Graphical): Dispersion metrics
dispersion_metrics = aggregated_data.describe()
print("Dispersion Metrics:")
print(dispersion_metrics)

# Univariate Analysis (Graphical)
def plot_histogram(column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(aggregated_data[column], kde=True, bins=30)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

plot_histogram('total_duration', 'Distribution of Total Duration')
plot_histogram('total_data_volume', 'Distribution of Total Data Volume')

# Bivariate Analysis
sns.pairplot(aggregated_data[['total_download', 'total_upload', 'total_data_volume']])
plt.show()

# Correlation Analysis
correlation_columns = ['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'others']
correlation_matrix = df[correlation_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Dimensionality Reduction (PCA)
features = ['total_duration', 'total_download', 'total_upload', 'total_data_volume']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(aggregated_data[features])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

print("PCA Explained Variance:")
print(explained_variance)

aggregated_data['PC1'] = principal_components[:, 0]
aggregated_data['PC2'] = principal_components[:, 1]

sns.scatterplot(x='PC1', y='PC2', data=aggregated_data)
plt.title('PCA: User Data')
plt.show()

# PCA Interpretation:
print("1. The first principal component explains the majority of variance.")
print("2. PCA helps identify patterns in user data by reducing dimensionality.")
print("3. Outliers in PCA plot can indicate unusual user behavior.")
print("4. Marketing campaigns can target clusters identified in PCA scatter plot.")
