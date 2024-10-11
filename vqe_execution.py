"""
     Computing a Ramsey Number using a Variational Quantum Eigensolver
"""

import sys

import ramsey_util as ru
import numpy as np

from qiskit_aer import AerSimulator, StatevectorSimulator 
from qiskit_aer.primitives import EstimatorV2, SamplerV2, Estimator
import qiskit_algorithms as qka
import qiskit as qk
import qiskit.quantum_info as qki
import qiskit.circuit.library as qkl

np.set_printoptions(linewidth=300)
data_csv_path = "./raw_data/vqe_execution_results.csv"

graph_order = 1
clique_order, iset_order = 2, 2
num_trials = 1
if len(sys.argv) == 5:
    graph_order = int(sys.argv[1])
    clique_order, iset_order = int(sys.argv[2]), int(sys.argv[3])
    num_trials = int(sys.argv[4])

edge_length = graph_order * (graph_order - 1) // 2

simulator = StatevectorSimulator(device="GPU")
estimator = Estimator()

print(f"Computing R({clique_order},{iset_order}) ?= {graph_order}")

print(f"{"":=<80}")
print(f"Graph Order: {graph_order}")
print(f"Clique Order: {clique_order}")
print(f"ISet Order: {iset_order}")

print(f"{"":-<80}")
print(f"Number of qubits: {edge_length}")
print(f"Number of trials: {num_trials}")
print(f"Output file: {data_csv_path}")

spsa = qka.optimizers.SPSA(maxiter=300)
problem_hamiltonian = ru.generate_ramsey_hamiltonian(graph_order, clique_order, iset_order)

data_csv = open(data_csv_path, "a+")

print(f"{"":-<80}")
for trial_i in range(num_trials):
    print(f"\rExecuting trial {trial_i + 1}", end="")
    ansatz = qkl.EfficientSU2(edge_length)
    vqe = qka.VQE(estimator, ansatz, optimizer=spsa)
    npe = qka.NumPyMinimumEigensolver()
    initial_point = np.random.random(ansatz.num_parameters)

    result = vqe.compute_minimum_eigenvalue(operator=problem_hamiltonian)
    numpy_result = npe.compute_minimum_eigenvalue(operator=problem_hamiltonian)

    optimal_circuit = result.optimal_circuit
    optimal_params = result.optimal_parameters
    optimal_circuit.assign_parameters(optimal_params, inplace=True)

    numpy_min_energy = np.float64(numpy_result.eigenvalue)
    min_energy = result.eigenvalue
    min_energy_state = qki.Statevector(optimal_circuit).data

    state_str = ",".join([str(x) for x in min_energy_state])
    data_csv.write(f"{graph_order};{clique_order};{iset_order};{numpy_min_energy};{min_energy};{state_str}\n")

data_csv.close()
print(f"\rCompleted Trials: results stored in {data_csv_path}")
