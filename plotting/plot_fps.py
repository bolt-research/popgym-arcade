import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# import sys; sys.path.extend(['/home/ubuntu-user/popjym-main/'])

alldata = pd.read_csv(
    "F:/Desktop/ML/Popjym/popjym/popjym/popjym/plotting/fps_data/all.csv"
)
popgymarcadedata = pd.read_csv(
    "F:/Desktop/ML/Popjym/popjym/popjym/popjym/plotting/fps_data/True_popgymarcadefpsdata.csv"
)

sns.set()
# sns.color_palette("Paired")

# plt.figure(figsize=(12, 8))
fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharex=True, sharey=True)

sns.lineplot(
    data=popgymarcadedata,
    x="Num Envs",
    y="FPS",
    hue="Environment",
    marker="o",
    markersize=25,
    ax=axes[0],
    # errorbar=('ci', 95)
)
axes[0].set_xscale("log", base=2)
axes[0].set_yscale("log", base=10)
axes[0].tick_params(axis="both", which="major", labelsize=35)
axes[0].legend(title="", title_fontsize="20", fontsize="22", ncol=3)
axes[0].set_xlabel("")
axes[0].set_ylabel("")
sns.lineplot(
    data=alldata,
    x="Num Envs",
    y="FPS",
    hue="Environment",
    marker="o",
    markersize=25,
    # errorbar=('ci', 95)
    ax=axes[1],
)

axes[1].set_xscale("log", base=2)
axes[1].set_yscale("log", base=2)
axes[1].tick_params(axis="both", which="major", labelsize=35)
axes[1].legend(title="", title_fontsize="20", fontsize="22", ncol=3)
axes[1].set_xlabel("")
axes[1].set_ylabel("")
fig.text(0.55, 0.04, "Number of Parallel Environments", ha="center", fontsize=35)
fig.text(0.04, 0.6, "Frames per second", va="center", rotation="vertical", fontsize=35)

# fig.subplots_adjust(wspace=0.3)

# fig.subplots_adjust(left=0.05, wspace=0.3)
plt.tight_layout(rect=[0.07, 0.07, 1, 1])
plt.show()

# plt.xlabel('Number of Parallel Environments', fontsize=50)
# plt.ylabel('Frames per second (FPS)', fontsize=50)
# # Optionally, adjust tick label sizes
# plt.tick_params(axis='both', which='major', labelsize=50)

# plt.xscale('log', base=2)
# plt.yscale('log', base=10)

# # Show the plot
# plt.legend(title='', title_fontsize='20', fontsize='20')
# plt.tight_layout()
# # plt.savefig('POMDP.png')
# plt.show()
