import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

alldata = pd.read_csv('./all.csv')
popgymarcadedata = pd.read_csv('./128fpsdata.csv')

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

colors = ['#f394c4', '#f2d580', '#b2b2b2', '#b2d28e', '#ecaf81', '#bab8d9', '#8dcfbb']
sns.set_palette(colors)

sns.lineplot(
    data=popgymarcadedata, 
    x='Num Envs',
    y='FPS', 
    hue='Environment',
    marker='o',
    markersize=25,
    ax=axes[0],
)

axes[0].set_xscale('log', base=2)
axes[0].set_yscale('log', base=10)
axes[0].tick_params(axis='both', which='major', labelsize=20)
axes[0].set_xlabel('')
axes[0].set_ylabel('')

colors = ['#f394c4', '#ecaf81', '#b2b2b2', '#b2d28e', '#8600ec', '#bab8d9', '#8dcfbb']
for i, env in enumerate(alldata['Environment'].unique()):
    env_data = alldata[alldata['Environment'] == env]
    marker_style = '*' if 'popgym arcade' in env.lower() else 'o'
    color = colors[i % len(colors)]
    
    sns.lineplot(
        data=env_data, 
        x='Num Envs',
        y='FPS', 
        label=env,
        marker=marker_style,
        markersize=25 if marker_style == '*' else 20,
        color=color,
        ax=axes[1],
    )


axes[1].set_xscale('log', base=2)
axes[1].set_yscale('log', base=10)
axes[1].tick_params(axis='both', which='major', labelsize=20)



axes[1].set_xlabel('')
axes[1].set_ylabel('')


axes[0].set_yticks([10000, 1000000, 100000000])
axes[1].set_yticks([10000, 1000000, 100000000])

axes[0].legend(title='', fontsize=12)
axes[1].legend(title='', fontsize=12)

fig.text(0.5, 0.02, 'Number of Parallel Environments', ha='center', fontsize=20)
fig.text(0.04, 0.6, 'Frames per second', va='center', rotation='vertical', fontsize=20)


for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15, left=0.1, wspace=0.1)
# plt.show()
plt.savefig("fps.pdf")