import pandas as pd

# Creating a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['Delhi', 'Mumbai', 'Bangalore', 'Chennai'],
    'Score': [85, 90, 95, 88]
}

df = pd.DataFrame(data)

# Display first few rows
print(df.head())

# Shape of DataFrame
print("Shape:", df.shape)

# Summary statistics
print(df.describe())

# Column names
print("Columns:", df.columns.tolist())

# Data types
print("Data types:\n", df.dtypes)

# Selecting a column
print(df['Name'])

# Filtering rows
print(df[df['Age'] > 30])

# Adding a new column
df['Passed'] = df['Score'] > 85
print(df)
