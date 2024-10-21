import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


fragments_df = pd.read_csv('fragments.csv')

# Compute correlation matrix
correlation_matrix = fragments_df[['Molecular Weight', 'LogP', 'Num H Donors', 'Num H Acceptors', 'AR', 'Predicted']].corr()

# Plot the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap between Fragment Properties and Predicted vs Actual (AR)')
plt.show()
