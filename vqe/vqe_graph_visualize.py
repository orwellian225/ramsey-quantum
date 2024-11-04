import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import polars as pl
import numpy as np
import math as m

mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white' 
mpl.rcParams['axes.edgecolor'] = 'white' 
mpl.rcParams['ytick.color'] = 'white'

data = pl.read_csv("./raw_data/vqe_execution_results.csv", separator=";")
data = data.with_columns(
    pl.col("*").exclude("minimum_energy_state"),
    pl.col("minimum_energy_state").str.split(",")
    .list.eval(
        pl.element().str.replace_all("\(|\)", "")
        ).map_elements(lambda arr: np.array(arr, dtype=np.complex128), return_dtype=pl.Object)
)

def mean_arrays(arrays):
    stack = np.stack(arrays)
    result = stack.mean(axis=0)
    return result

result = data.group_by("graph_order", "clique_order", "iset_order").agg(
    pl.len().alias("num_trials"),
    pl.col("minimum_energy").mean(),
    pl.col("numpy_minimum_energy").mean(),
    # pl.col("minimum_energy_state").map_elements(mean_arrays, return_dtype=pl.Object),
    pl.col("minimum_energy_state")
).sort(pl.col("graph_order"))


for i, row in enumerate(result.iter_rows()):
    energy_idx = 0 
    state_idx = 1
    graph_order, clique_order, iset_order, num_trials, npe_min_energy, vqe_min_energy, vqe_min_state = row
    edge_length = graph_order * (graph_order - 1) // 2

    vqe_min_state = np.array(vqe_min_state[1])
    vqe_probs = vqe_min_state[:].real**2 + vqe_min_state[:].imag**2
    eigenkets = [ bin(i)[2:].zfill(edge_length) for i in range(2**edge_length) ]

    plt.bar(x=eigenkets, height=(np.abs(vqe_min_state)**2), color=sns.color_palette("Set2")[0])
    break

plt.gca().patch.set_facecolor('#1b1b1b')
plt.gcf().patch.set_facecolor('#1b1b1b')
plt.savefig("./visualizations/vqe_probability_distribution.pdf")