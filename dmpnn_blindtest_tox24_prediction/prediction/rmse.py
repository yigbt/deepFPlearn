import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import sys

# Load the merged dataset (after combining actual and predicted values)
merged_df = pd.read_csv(sys.argv[1])

# Extract the 'actual' and 'predicted' columns
actual_values = merged_df['actual']
predicted_values = merged_df['predicted']

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

# Print the RMSE
print(f"Root Mean Squared Error (RMSE): {rmse}")

