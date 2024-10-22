import pandas as pd
import matplotlib.pyplot as plt

# Load the merged dataset (after combining actual and predicted values)
merged_df = pd.read_csv('merged_actual_predicted.csv')

# Extract the 'actual' and 'predicted' columns
actual_values = merged_df['actual']
predicted_values = merged_df['predicted']

# Create the actual vs predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(actual_values, predicted_values, color='blue', alpha=0.6)
plt.plot([-60, 160], [-60,160], color='red', linestyle='--', linewidth=2)
plt.xlim(-60, 160)
plt.ylim(-60, 160)

# Add a line for perfect predictions (y = x)
#plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], color='red', linestyle='--', linewidth=2)

# Add labels and title
plt.title("Actual vs Predicted dmpnn tox24 test")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

# Display the plot
plt.show()
