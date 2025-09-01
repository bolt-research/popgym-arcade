import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set a professional plot style
#sns.set_theme(style="whitegrid", context="talk")
sns.set()

# 1. Load the data from the provided CSV structure
# Using the original io.StringIO for reproducibility, but this works with your file path.
df = pd.read_csv("/Users/smorad/Downloads/finalpqn128model_group.csv")
# Drop FART
df = df[df["Algorithm"] != 'PQN_RNN (Fart)']

# Clean up algorithm names
df['Algorithm'] = df['Algorithm'].str.replace('PQN_RNN \(', '', regex=True).str.replace('\)', '', regex=True)
df['Algorithm'] = df['Algorithm'].str.replace('PQN (MLP', 'MLP (Baseline)')

# 2. Calculate derived metrics and propagate error
df['sem'] = df['std'] / np.sqrt(df['count'])

pivoted = df.pivot_table(
    index='Algorithm', 
    columns='Partial', 
    values=['mean', 'sem']
)
pivoted.columns = [f'{val}_{col}' for val, col in pivoted.columns]
pivoted.reset_index(inplace=True)

# --- Calculate Observability Gap ---
pivoted['gap'] = pivoted['mean_False'] - pivoted['mean_True']
pivoted['gap_sem'] = np.sqrt(pivoted['sem_False']**2 + pivoted['sem_True']**2)

# --- Calculate Memory Bias ---
# Bias = J(pi, M) - J(f, pi, M). Positive bias = recurrent model is worse on MDP.
mlp_mdp_mean = pivoted.loc[pivoted['Algorithm'] == 'MLP (Baseline)', 'mean_False'].iloc[0]
mlp_mdp_sem = pivoted.loc[pivoted['Algorithm'] == 'MLP (Baseline)', 'sem_False'].iloc[0]

pivoted['bias'] = pivoted['mean_False'] - mlp_mdp_mean
pivoted['bias_sem'] = np.sqrt(mlp_mdp_sem**2 + pivoted['sem_False']**2)

# By definition, the bias of the baseline against itself is zero.
mlp_index = pivoted[pivoted['Algorithm'] == 'MLP (Baseline)'].index
pivoted.loc[mlp_index, 'bias'] = 0
pivoted.loc[mlp_index, 'bias_sem'] = 0

# Define a logical order for the plot and sort the data
algo_order = ['MLP (Baseline)', 'Fart', 'Mingru', 'Lru']
pivoted['Algorithm'] = pd.Categorical(pivoted['Algorithm'], categories=algo_order, ordered=True)
plot_data = pivoted.sort_values('Algorithm')

# 3. Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=False)
algorithms = plot_data['Algorithm']
colors = sns.color_palette('colorblind', n_colors=len(algorithms))

# --- Plot 1: Return (on POMDP) ---
ax1 = axes[0]
ax1.bar(
    algorithms,
    plot_data['mean_True'],
    yerr=plot_data['sem_True'],
    capsize=5,
    color=colors,
    alpha=0.8
)
ax1.set_title('Return ($J(f, \pi, \mathcal{P})$)')
ax1.set_ylabel('Mean Return')
ax1.set_ylim(bottom=0)

# --- Plot 2: Observability Gap ---
ax2 = axes[1]
ax2.bar(
    algorithms,
    plot_data['gap'],
    yerr=plot_data['gap_sem'],
    capsize=5,
    color=colors,
    alpha=0.8
)
ax2.set_title('Observability Gap')
ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')

# --- Plot 3: Memory Bias ---
ax3 = axes[2]
ax3.bar(
    algorithms,
    plot_data['bias'],
    yerr=plot_data['bias_sem'],
    capsize=5,
    color=colors,
    alpha=0.8
)
ax3.set_title('Memory Bias')
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Final adjustments
for ax in axes:
    ax.set_xlabel('Memory Model')
    #ax.tick_params(axis='x', rotation=45, ha='right')

fig.suptitle('Disentangling Memory and Policy Performance', fontsize=20, y=1.03)
plt.tight_layout()
plt.show()
