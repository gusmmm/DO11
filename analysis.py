import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
# Note: You'll need to specify the path to your CSV file
df = pd.read_csv('Key_indicator_districtwise.csv')  # Replace with your actual CSV filename

# Display the first few rows of the dataframe
print("First 5 rows of the dataframe:")
print(df.head())

# Display information about the dataframe
print("\nDataframe information:")
print(df.info())

# Basic statistics
print("\nBasic statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values)
print("\nPercentage of missing values per column:")
print(100 * missing_values / len(df), "%")

# Check data types distribution
print("\nData types distribution:")
print(df.dtypes.value_counts())

# Potential outliers detection (using Z-score for numerical columns)
print("\nPotential outliers by column (Z-score > 3):")
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
    outliers_count = (z_scores > 3).sum()
    if outliers_count > 0:
        print(f"{col}: {outliers_count} potential outliers")

# Distribution plots for numeric columns
print("\nCreating distribution plots for numerical columns...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:min(9, len(numeric_cols))]):  # Limit to 9 columns for visualization
    plt.subplot(3, 3, i+1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig('numeric_distributions.png')

# Categorical data analysis
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print("\nCategorical column value counts:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head(10))  # Show top 10 values

# Correlation matrix for numeric data
# if len(numeric_cols) > 1:
#     print("\nGenerating correlation matrix...")
#     plt.figure(figsize=(12, 10))
#     correlation = df[numeric_cols].corr()
#     sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
#     plt.title('Correlation Matrix')
#     plt.tight_layout()
#     plt.savefig('correlation_matrix.png')
#     plt.show()  # Add this line to display the plot

#     # Optional: Filter correlation matrix to show only strong correlations
#     print("\nStrong correlations (absolute value > 0.5):")
#     strong_corr = correlation.unstack()
#     strong_corr = strong_corr[strong_corr.abs() > 0.5]
#     strong_corr = strong_corr[strong_corr != 1.0]  # Remove self-correlations
#     if not strong_corr.empty:
#         print(strong_corr.sort_values(ascending=False))

print("\nExploratory Data Analysis completed!")