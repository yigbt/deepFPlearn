import pandas as pd
import matplotlib.pyplot as plt

# Define file paths (files need to be uploaded to proceed)
train_file_path = 'dfpl_train_tox24__leaderboard_predictions.csv'
test_file_path = 'dfpl_test_tox24_predictions.csv'

# Attempting to load the data files

train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Set the figure size
plt.figure(figsize=(10, 10))

# Plot training data in blue
plt.scatter(train_data['actual'], train_data['predicted'], color='blue', alpha=0.6, label='Training Data')

# Plot test data in yellow
plt.scatter(test_data['actual'], test_data['predicted'], color='yellow', alpha=0.6, label='Test Data')

# Add y=x reference line in red
min_val = min(train_data['actual'].min(), test_data['actual'].min(), train_data['predicted'].min(),
              test_data['predicted'].min())
max_val = max(train_data['actual'].max(), test_data['actual'].max(), train_data['predicted'].max(),
              test_data['predicted'].max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

# Set equal limits for x and y axes
plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.gca().set_aspect('equal', adjustable='box')

# Add labels and legend
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend()

# Show the plot
plt.show()



