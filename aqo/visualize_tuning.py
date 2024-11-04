import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import polars as pl
import numpy as np

k, l = 2, 4
max_time_space = [1., 2., 3.]
df = pl.read_csv(f"./raw_data/final_adiabatic_tuning_R({k},{l}).csv")

fig, axes = plt.subplots(ncols=len(max_time_space), figsize=(len(max_time_space) * 5, 5))
# fig.suptitle(f"Evaluating AQO performance at various $T$ and $M$ for R({k},{l})")

for i, t in enumerate(max_time_space):
    _ = sns.lineplot(data=df.filter(pl.col("max_t") == t), hue="graph_order", x="delta", y="estimated_energy", palette="Set2", ax=axes[i], sort=True)
    for n in range(3, df.max().item(row=0, column="graph_order") + 1):

        # print(df.filter(
        #     (pl.col("max_t") == t) & (pl.col("graph_order") == n)
        # ).min().item(row=0, column="actual_energy"))

        axes[i].axhline(y=df.filter(
            (pl.col("max_t") == t) & (pl.col("graph_order") == n)
        ).min().item(row=0, column="actual_energy"), color=sns.color_palette("Set2")[n - 3], linestyle="--")
    axes[i].set_yticks(np.concat([np.array([0.5]), np.arange(0, 5)]))
    axes[i].set_xticks(np.array([1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01]))
    axes[i].set_xscale('log')
    _ = axes[i].set_title(f'Estimated Energy vs $\Delta t$ at $T={t}$')
    legend = axes[i].legend(frameon = 1, title=fr'$R({k},{l})$ =?')
    frame = legend.get_frame()


plt.savefig(f"./visualizations/final_adibatic_tuning_R({k}_{l}).pdf")