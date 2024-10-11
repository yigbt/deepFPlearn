import pandas as pd
import json

# Load the CSV file
df = pd.read_csv('regression_cytotox.csv')

# Extract the 'AR' column (for normalization)
target_column = 'AR'
min_val = df[target_column].min()
max_val = df[target_column].max()

# Apply Min-Max normalization
#df['AR_normalized'] = (df[target_column] - min_val) / (max_val - min_val) # between 0 and 1
df['AR_normalized'] = 2 * ((df[target_column] - min_val) / (max_val - min_val)) - 1 # between -1 and 1
# Save the normalized values to a new CSV
df.to_csv('normalized_output.csv', index=False)

# Print the min and max values for denormalization
print(f"Min value: {min_val}")
print(f"Max value: {max_val}")

min_max_values = {'min': min_val, 'max': max_val}
with open('min_max_values.json', 'w') as f:
    json.dump(min_max_values, f)






# Load the normalized CSV file
df2 = pd.read_csv('AR_single-labeled_Fold-4.y_test.csv')


# Denormalize the values
#df2['denormalized'] = df2['0'].apply(lambda x: x * (max_val - min_val) + min_val)
df2['denormalized'] = df2['0'].apply(lambda x: (x + 1) / 2 * (max_val - min_val) + min_val)
# Save the denormalized values to a new CSV
df2.to_csv('denormalized_output.csv', index=False)














