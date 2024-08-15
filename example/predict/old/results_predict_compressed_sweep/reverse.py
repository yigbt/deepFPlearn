import pandas as pd
from sklearn.preprocessing import StandardScaler
import json

df_scaled = pd.read_csv('predict_data_AR.csv')

with open('scaling_params.json', 'r') as f:
    scaling_params = json.load(f)

# Reverse the scaling using the saved mean and scale
df_scaled['AR'] = df_scaled['AR'] * scaling_params['scale'] + scaling_params['mean']
df_scaled['predicted'] = df_scaled['predicted'] * scaling_params['scale'] + scaling_params['mean']

# Save the reversed data to a new CSV file
df_scaled.to_csv('reversed_data.csv', index=False)

print("Reversed scaling completed and saved to 'reversed_data.csv'.")

