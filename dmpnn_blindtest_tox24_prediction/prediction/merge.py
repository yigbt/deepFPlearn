import pandas as pd

# Load the actual and predicted data files
actual_df = pd.read_csv('tox24_blindtest_nonscaled.csv')
predicted_df = pd.read_csv('dmpnn_test_tox24_predictions.csv')

# Clean the 'smiles' column in the predicted dataframe by removing the brackets and quotes
predicted_df['smiles'] = predicted_df['smiles'].str.replace(r"[\[\]']", "", regex=True)

# Merge the two dataframes on the 'SMILES' and 'smiles' columns
merged_df = pd.merge(actual_df, predicted_df, left_on='SMILES', right_on='smiles', how='inner')

# Drop the extra 'smiles' column from the merged dataframe
merged_df = merged_df.drop(columns=['smiles'])

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_actual_predicted.csv', index=False)

print("Merged file saved as 'merged_actual_predicted.csv'")
