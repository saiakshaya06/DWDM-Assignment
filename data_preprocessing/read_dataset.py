
import pandas as pd

# Load the Kaggle dataset
df = pd.read_csv("dataset/mushrooms.csv")

# Display first 5 records
print("\nFIRST 5 RECORDS:")
print(df.head())

# Display dataset size
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display information about the dataset
print("\nDATASET INFORMATION:")
print(df.info())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

# Display edible and poisonous mushroom counts
print("\nCLASS DISTRIBUTION:")
print(df["class"].value_counts())
