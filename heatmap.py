import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Load your data from the CSV file
data = pd.read_csv('sensitivity_analysis_results.csv')

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Convert DidNotComplete column to boolean
data['DidNotComplete'] = data['DidNotComplete'].apply(lambda x: str(x).strip().lower() == 'true')

# Get the unique parameters in the order they appear in the CSV
parameters_order = data['Parameter'].unique()

# Pivot the data to get the structure needed for the heatmap
pivot_table = data.pivot(index='FoldChange', columns='Parameter', values='ViralCycleTime')

# Ensure the columns are in the specified order
pivot_table = pivot_table[parameters_order]

# Generate the mask for "Did not complete cycle"
mask = data.pivot(index='FoldChange', columns='Parameter', values='DidNotComplete').astype(bool)
mask = mask[parameters_order]

# Define a custom colormap to match the provided image
colors = ["#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FFA500", "#FF0000"]
cmap = LinearSegmentedColormap.from_list("custom_blue_green_yellow", colors, N=256)

# Define custom boundaries and normalization for the color scale
boundaries = [1.6, 2.5, 4, 6, 10, 16, 25, 30]
norm = BoundaryNorm(boundaries, cmap.N, clip=True)

# Create the heatmap
plt.figure(figsize=(14, 10))

# Plot the heatmap with custom colormap and normalization
ax = sns.heatmap(pivot_table, annot=False, mask=mask, cmap=cmap, norm=norm, cbar_kws={'label': 'Viral cycle time (hr)', 'ticks': boundaries})

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
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color='black', ec='black'))
            ax.text(j + 0.5, i + 0.5, 'Did not\ncomplete\ncycle', ha='center', va='center', color='white', fontsize=8, weight='bold')

plt.tight_layout()
plt.show()