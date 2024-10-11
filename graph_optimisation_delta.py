"""
    Graphing the Trotterization Delta created for testing an AQO problem
    
    Plotting Eigenvalues (Energy) against delta 
"""

import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import polars as pl
import numpy as np
import math as m

from adiabatic_optimisation import AdiabaticOptimisationProblem
import qiskit as qk
import qiskit_algorithms as qka
from qiskit_aer import AerSimulator, StatevectorSimulator, AerError
from qiskit_aer.primitives import EstimatorV2

k = 3
l = 3
max_n = 5 + 1
initial_hams = [ru.generate_initial_hamiltonian(n) for n in range(2, max_n)]
problem_hams = [ru.generate_ramsey_hamiltonian(n, k, l) for n in range(2, max_n)]
# min_energies = [np.min(np.linalg.eigvals(h)).astype(np.float32) for h in problem_hams]
np_min_energies = []
max_time_space = [.5, 1., 2., 3., 4., 5.]
deltas_space = np.linspace(0., 1., 50)[1:]

data = {
    "max_t": [],
    "steps": [],
    "delta": [],
    "graph_order": [],
    "estimated_energy": [],
}

sim = StatevectorSimulator(device='GPU')
estimator = EstimatorV2()

for n in range(2, max_n):

    npe = qka.NumPyMinimumEigensolver()
    npe_result = npe.compute_minimum_eigenvalue(operator=problem_hams[n - 2])
    npe_min_energy = npe_result.eigenvalue
    np_min_energies.append(npe_min_energy)

    for T in max_time_space:
        for delta in deltas_space:
            print(f"\rSimulation R({k},{l}) ?= {n} for T = {T}, Steps = {m.ceil(T / delta)} & delta = {delta} ", end="")

            aqo = AdiabaticOptimisationProblem(
                initial_hamiltonian=initial_hams[n - 2],
                problem_hamiltonian=problem_hams[n - 2],
                time=T,
                steps= int(m.ceil(T / delta))
            )
            aqo.generate([1 / m.sqrt(2**aqo.num_qubits) for _ in range(2**aqo.num_qubits)])
            aqo.circuit = aqo.circuit.decompose()

            observables = [aqo.problem_hamiltonian]
            pub = (qk.transpile(aqo.circuit, sim), observables)
            job = estimator.run([pub])

            data["max_t"].append(T)
            data["steps"].append(int(T // delta))
            data["delta"].append(delta)
            data["graph_order"].append(n)
            data["estimated_energy"].append(job.result()[0].data.evs[0])
print("")

df = pl.from_dict(data)
fig, axes = plt.subplots(ncols=len(max_time_space), figsize=(len(max_time_space) * 5, 5))
fig.suptitle(f"Evaluating AQO performance at various $T$ and $M$ for R({k},{l})")

for i, t in enumerate(max_time_space):
    _ = sns.lineplot(data=df.filter(pl.col("max_t") == t), hue="graph_order", x="delta", y="estimated_energy", palette="Set2", ax=axes[i], sort=True)
    for n in range(2, max_n):
        axes[i].axhline(y=np_min_energies[n - 2], color=sns.color_palette("Set2")[n - 2], linestyle="--")
    _ = axes[i].set_title(f'Estimated Energy vs $\Delta t$ at $T={t}$')

plt.savefig(f"./visualizations/adibatic_tuning_R({k}_{l}).pdf")
