import pandas as pd
import matplotlib.pyplot as plt

# Load the merged dataset (after combining actual and predicted values)
df = pd.read_csv('dfpl_test_tox24_predictions.csv')

# Extract the 'actual' and 'predicted' columns
actual_values = df['actual']
predicted_values = df['predicted']

# Create the actual vs predicted plot
plt.figure(figsize=(8, 6))
plt.scatter(actual_values, predicted_values, color='blue', alpha=0.6)

# Add a line for perfect predictions (y = x)
plt.plot([-60, 160], [-60,160], color='red', linestyle='--', linewidth=2)
plt.xlim(-60, 160)
plt.ylim(-60, 160)

# Add labels and title
plt.title("Actual vs Predicted dfpl-fp tox24 test")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

# Display the plot
plt.show()
