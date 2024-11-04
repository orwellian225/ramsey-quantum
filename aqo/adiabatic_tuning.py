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
from qiskit_aer import StatevectorSimulator
from qiskit_aer.primitives import EstimatorV2

k = 2
l = 4
max_n = 5 + 1
initial_hams = [ru.generate_initial_hamiltonian(n) for n in range(2, max_n)]
problem_hams = [ru.generate_ramsey_hamiltonian(n, k, l) for n in range(2, max_n)]
# min_energies = [np.min(np.linalg.eigvals(h)).astype(np.float32) for h in problem_hams]
np_min_energies = []
max_time_space = [1., 2., 3.]
deltas_space= np.array([1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.01])

data = {
    "max_t": [],
    "steps": [],
    "delta": [],
    "graph_order": [],
    "actual_energy": [],
    "estimated_energy": [],
}

sim = StatevectorSimulator()
estimator = EstimatorV2()

for n in range(3, max_n):

    npe = qka.NumPyMinimumEigensolver()
    npe_result = npe.compute_minimum_eigenvalue(operator=problem_hams[n - 2])
    npe_min_energy = npe_result.eigenvalue
    np_min_energies.append(npe_min_energy)

    for T in max_time_space:
        for i, delta in enumerate(deltas_space):
            steps = int(T / delta)
            print(f"Simulation R({k},{l}) ?= {n} for T = {T}, Steps = {steps} & delta = {delta:.2f}")

            aqo = AdiabaticOptimisationProblem(
                initial_hamiltonian=initial_hams[n - 2],
                problem_hamiltonian=problem_hams[n - 2],
                time=T,
                steps=steps
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
            data["actual_energy"].append(npe_min_energy)
            data["estimated_energy"].append(job.result()[0].data.evs[0])
print("")

df = pl.from_dict(data)
print(df)
df.write_csv(f"./raw_data/final_adiabatic_tuning_R({k},{l}).csv")