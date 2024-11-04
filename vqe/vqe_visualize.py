import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import math as m

data = pl.read_csv("./raw_data/vqe_execution_results.csv", separator=";")
data = data.with_columns(
    pl.col("*").exclude("minimum_energy_state"),
    pl.col("minimum_energy_state").str.split(",")
    .list.eval(
        pl.element().str.replace_all("\(|\)", "")
        ).map_elements(lambda arr: np.array(arr, dtype=np.complex128), return_dtype=pl.Object)
)
# Converting the statevector into a numpy array is the above squiggles

print(data)

def mean_arrays(arrays):
    stack = np.stack(arrays)
    result = stack.mean(axis=0)
    return result

result = data.group_by("graph_order", "clique_order", "iset_order").agg(
    pl.len().alias("num_trials"),
    pl.col("minimum_energy").mean(),
    pl.col("numpy_minimum_energy").mean(),
    pl.col("minimum_energy_state").map_elements(mean_arrays, return_dtype=pl.Object),
).sort(pl.col("graph_order"))
print(result)

plot_index = lambda i: (i % len(result) // 2, 2 * (i % 2), 2 * (i % 2) + 1)
# fig, axes = plt.subplots(nrows=int(m.ceil(len(result) / 2)), ncols=4, figsize=(16, int(4 * m.ceil(len(result) / 2))))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))

# for r in range(int(m.ceil(len(result) / 2))):
#     for c in range(4):
        # axes[r, c].axis('off')
for c in range(2):
    axes[c].axis('off')

for i, row in enumerate(result.iter_rows()):
    # row_idx, energy_idx, state_idx = plot_index(i)
    energy_idx = 0 
    state_idx = 1
    graph_order, clique_order, iset_order, num_trials, npe_min_energy, vqe_min_energy, vqe_min_state = row
    edge_length = graph_order * (graph_order - 1) // 2

    vqe_probabilites = np.abs(vqe_min_state)**2
    eigenkets = [ bin(i)[2:].zfill(edge_length) for i in range(2**edge_length) ]

    # axes[row_idx, energy_idx].text(x=0., y=0.9, s=f"Graph Order: {graph_order}")
    # axes[row_idx, energy_idx].text(x=0., y=0.825, s=f"Clique Order: {clique_order}")
    # axes[row_idx, energy_idx].text(x=0., y=0.75, s=f"ISet Order: {iset_order}")
    # axes[row_idx, energy_idx].text(x=0., y=0.675, s=f"Number of trials: {num_trials}")
    # axes[row_idx, energy_idx].text(x=0., y=0.5, s=f"Numpy Computed Mean Energy: {npe_min_energy:.2f}")
    # axes[row_idx, energy_idx].text(x=0., y=0.425, s=f"VQE Computed Mean Energy: {vqe_min_energy:.2f}")
    # axes[row_idx, energy_idx].text(x=0., y=0.35, s=f"$\Delta$ Energy: {npe_min_energy - vqe_min_energy:.2f}")
    # axes[row_idx, energy_idx].text(x=0., y=0.275, s=f"Most likely state: {eigenkets[np.argmax(vqe_probabilites)]}")
    # axes[row_idx, energy_idx].text(x=0., y=0.2, s=f"Most likely state probability: {np.max(vqe_probabilites):.1E}")
    # axes[row_idx, energy_idx].set_title(f"R({clique_order},{iset_order}) ?< {graph_order} Info")

    # axes[row_idx, state_idx].set_title(f"R({clique_order},{iset_order}) ?< {graph_order} Statevector Distribution")
    # axes[row_idx, state_idx].axis('on')

    # axes[row_idx, state_idx].bar(x=eigenkets, height=(np.abs(vqe_min_state)**2))
    # if i > 0:
    #     axes[row_idx, state_idx].set_xticks([])
    axes[energy_idx].text(x=0., y=0.9, s=f"Graph Order: {graph_order}")
    axes[energy_idx].text(x=0., y=0.825, s=f"Clique Order: {clique_order}")
    axes[energy_idx].text(x=0., y=0.75, s=f"ISet Order: {iset_order}")
    axes[energy_idx].text(x=0., y=0.675, s=f"Number of trials: {num_trials}")
    axes[energy_idx].text(x=0., y=0.5, s=f"Numpy Computed Mean Energy: {npe_min_energy:.2f}")
    axes[energy_idx].text(x=0., y=0.425, s=f"VQE Computed Mean Energy: {vqe_min_energy:.2f}")
    axes[energy_idx].text(x=0., y=0.35, s=f"$\Delta$ Energy: {npe_min_energy - vqe_min_energy:.2f}")
    axes[energy_idx].text(x=0., y=0.275, s=f"Most likely state: {eigenkets[np.argmax(vqe_probabilites)]}")
    axes[energy_idx].text(x=0., y=0.2, s=f"Most likely state probability: {np.max(vqe_probabilites):.1E}")
    axes[energy_idx].set_title(f"R({clique_order},{iset_order}) ?< {graph_order} Info")

    axes[state_idx].set_title(f"R({clique_order},{iset_order}) ?< {graph_order} Statevector Distribution")
    axes[state_idx].axis('on')

    axes[state_idx].bar(x=eigenkets, height=(np.abs(vqe_min_state)**2))
    break
    if i > 0:
        axes[row_idx, state_idx].set_xticks([])


plt.suptitle("Computation Results of a Varitional Quantum Eigensolver on Ramsey Numbers")
plt.savefig("./visualizations/vqe_results_r33_3.pdf")
plt.show()

