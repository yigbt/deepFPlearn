import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the dataset from CSV
file_path = 'tox24_blindtest_nonscaled.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Define the min and max values based on the training data
min_value = -45
max_value = 111.12

# Initialize the MinMaxScaler with the desired feature range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# Apply the scaling to the "TTR binding activity" column
df[['TTR binding activity']] = scaler.fit_transform(df[['TTR binding activity']].apply(lambda x: (x - min_value) / (max_value - min_value)))

# Save the scaled data to a new CSV file
scaled_file_path = 'tox24_blindtest_scaled.csv'  # Name of the output file
df.to_csv(scaled_file_path, index=False)

print(f"Scaled dataset saved to {scaled_file_path}")

