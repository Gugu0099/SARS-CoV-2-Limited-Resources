import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Load your data from the CSV file
data = pd.read_csv('data.csv')

# Check the column types
print("Column Data Types:")
print(data.dtypes)

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Convert DidNotComplete column to boolean
data['DidNotComplete'] = data['DidNotComplete'].apply(lambda x: str(x).strip().lower() == 'true')

# Print the loaded data to verify
#print("\nLoaded Data:")
#print(data.head())

# Pivot the data to get the structure needed for the heatmap
pivot_table = data.pivot(index='FoldChange', columns='Parameter', values='ViralCycleTime')
#print("\nPivot Table:")
#print(pivot_table)  # Check the pivot table

# Generate the mask for "Did not complete cycle"
mask = data.pivot(index='FoldChange', columns='Parameter', values='DidNotComplete').astype(bool)
#print("\nMask:")
#print(mask)  # Check the mask

# Check if the mask contains both True and False values
#print("\nMask Value Counts:")
#print(mask.values)

# Create the heatmap
plt.figure(figsize=(12, 8))
ax = sns.heatmap(pivot_table, annot=False, mask=mask, cmap='viridis', cbar_kws={'label': 'Viral cycle time (hr)'})

# Customize the axes
ax.set_xticks(np.arange(len(pivot_table.columns)) + 0.5)
ax.set_xticklabels(pivot_table.columns, rotation=90)
ax.set_yticks(np.arange(len(pivot_table.index)) + 0.5)
ax.set_yticklabels(pivot_table.index)
ax.set_xlabel('Parameters')
ax.set_ylabel('Fold-change from base value')
plt.title('Sensitivity Analysis of Model Parameters')

# Adding text for black areas ("Did not complete cycle")
for i in range(pivot_table.shape[0]):
    for j in range(pivot_table.shape[1]):
        if mask.iloc[i, j]:
            ax.text(j + 0.5, i + 0.5, 'Did not\ncomplete\ncycle', ha='center', va='center', color='white', fontsize=8)

plt.tight_layout()
plt.show()