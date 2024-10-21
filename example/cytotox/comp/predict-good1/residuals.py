import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fragments_df = pd.read_csv('fragments.csv')

# Add residual column (Predicted - Actual)
fragments_df['Residual'] = fragments_df['Predicted'] - fragments_df['AR']

# Visualize the distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(fragments_df['Residual'], bins=30, kde=True)
#sns.boxplot(fragments_df['Residual'])
plt.title('Distribution of Residuals (Predicted - Actual)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

exit()
# Look at Residuals per functional group
plt.figure(figsize=(10, 6))
sns.boxplot(x='Functional Group', y='Residual', data=fragments_df)
plt.title('Residuals by Functional Group')
plt.xticks(rotation=45)
plt.show()
