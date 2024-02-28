import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib

plt.rcParams['font.family'] = 'Arial'

# Function to read data
def read_data(file_name, sheet_name):
    return pd.read_excel(file_name, sheet_name=sheet_name)

# Birch-Murnaghan EOS
def birch_murnaghan(V, V0, K0, K0_prime):
    eta = (V0 / V) ** (2 / 3)
    return 3 / 2 * K0 * (eta ** 7 - eta ** 5) * (1 + 3 / 4 * (K0_prime - 4) * (eta ** 2 - 1))

# Fit the Birch-Murnaghan EOS and return parameters
def fit_eos(P, V):
    params, cov = curve_fit(birch_murnaghan, V, P, p0=[np.mean(V), 4, 4])
    return params  # V0, K0, K0_prime

# Filter data based on temperature increments
def filter_by_temp_increment(df, increment):
    min_temp, max_temp = df['T'].min(), df['T'].max()
    temps_to_include = np.arange(min_temp, max_temp + increment, increment)
    return df[df['T'].isin(temps_to_include)]

file_name = 'file_name.xlsx'
fig, ax = plt.subplots(figsize=(10, 8))

# Adjust tick parameters for aesthetics
ax.tick_params(axis='both', which='both', top=True, right=True, direction='in', length=5, labelsize=18)

# Dataset information
datasets = [
    {'sheet_name': 'sheet_name1', 'cmap': 'winter', 'label': 'sheet_name1'},
    {'sheet_name': 'sheet_name2', 'cmap': 'autumn', 'label': 'sheet_name2'}
]

legend_positions = [(.81, 0.98), (.98, 0.98)]  # Pre-defined legend positions

for index, dataset in enumerate(datasets):
    df = read_data(file_name, dataset['sheet_name'])
    filtered_df = filter_by_temp_increment(df, 300)
    # Update to use the new recommended approach for accessing colormaps
    cmap = matplotlib.colormaps[dataset['cmap']]
    num_temps = len(filtered_df['T'].unique())

    handles = []  # List to store handles for the legend
    labels = []   # List to store labels for the legend

    # Plot each temperature group for the dataset and collect handles/labels
    for i, (T, group) in enumerate(filtered_df.groupby('T')):
        color = cmap(i / num_temps)
        line, = ax.plot(group['P'], group['V'], label=f'{T} K', color=color)
        handles.append(line)
        labels.append(f'{T} K')

    # Create legend for the current dataset with its specific handles and labels
    legend = ax.legend(handles, labels, title=dataset['label'], loc='upper right',
                       bbox_to_anchor=legend_positions[index], borderaxespad=0,
                       title_fontsize=15, fontsize=15)
    ax.add_artist(legend)  # Ensure this legend remains when adding the next

# Set axis labels with specified font size
plt.xlabel('Pressure (GPa)', fontsize=20)
plt.ylabel('Volume ($\AA^3$)', fontsize=20)
plt.tight_layout()
#plt.tight_layout(rect=[0, 0, 0.75, 1])  # Adjust the plot area to fit legends
plt.savefig("dft-pvt-all.jpg")
plt.show()
