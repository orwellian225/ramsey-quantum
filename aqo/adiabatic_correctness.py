import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import polars as pl
import numpy as np
import math as m

from adiabatic_optimisation import AdiabaticOptimisationProblem
import qiskit as qk
import qiskit_algorithms as qka
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

n_values = [4, 5, 6, 3, 4, 4, 5]
k_values = [3, 3, 3, 2, 2, 2, 2]
l_values = [3, 3, 3, 4, 4, 5, 5]
expected_energy = [0, 0, 1, 0, 1, 0, 1]

num_experiments = 7
T = 3
delta = 0.1
steps = int(T / delta)

sim = AerSimulator()
estimator = EstimatorV2()

print("Expected Energy Values:")
print("\t 0 - Expecting a zero energy")
print("\t 1 - Expecting a non-zero energy")
print("n, k, l, expected_energy, measured_energy")
for i in range(num_experiments):
    initial_ham = ru.generate_initial_hamiltonian(n_values[i])
    problem_ham = ru.generate_ramsey_hamiltonian(n_values[i], k_values[i], l_values[i])
    aqo = AdiabaticOptimisationProblem(
        initial_hamiltonian=initial_ham,
        problem_hamiltonian=problem_ham,
        time=T,
        steps=steps
    )

    aqo.generate([1 / m.sqrt(2**aqo.num_qubits) for _ in range(2**aqo.num_qubits)])
    aqo.circuit = aqo.circuit.decompose()

    observables = [aqo.problem_hamiltonian]
    pub = (qk.transpile(aqo.circuit, sim), observables)
    job = estimator.run([pub])
    measured_energy = job.result()[0].data.evs[0]
    print(f"{n_values[i]}, {k_values[i]}, {l_values[i]}, {expected_energy[i]}, {measured_energy}")
