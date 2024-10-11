"""
     Computing a Ramsey Number using a Variational Quantum Eigensolver
"""

import time
import sys
import os

import ramsey_util as ru
import numpy as np

import warnings

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator
from qiskit.primitives import StatevectorEstimator

import qiskit_algorithms as qka
import qiskit as qk
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import qiskit.quantum_info as qki
import qiskit.circuit.library as qkl

np.set_printoptions(linewidth=300)
warnings.filterwarnings('ignore', category=np.exceptions.ComplexWarning)
data_csv_path = "./raw_data/vqe_quantum_execution_results.csv"

use_sim = False

graph_order = 6
clique_order, iset_order = 2, 2
num_trials = 1
if len(sys.argv) == 5:
    graph_order = int(sys.argv[1])
    clique_order, iset_order = int(sys.argv[2]), int(sys.argv[3])
    num_trials = int(sys.argv[4])

edge_length = graph_order * (graph_order - 1) // 2

print(f"Computing R({clique_order},{iset_order}) ?= {graph_order}")

print(f"{"":=<80}")
print(f"Graph Order: {graph_order}")
print(f"Clique Order: {clique_order}")
print(f"ISet Order: {iset_order}")

print(f"{"":-<80}")
print(f"Number of qubits: {edge_length}")
print(f"Number of trials: {num_trials}")
print(f"Output file: {data_csv_path}")

print(f"{"":-<80}")
print(f"Connecting to IBM\r", end="")
service = QiskitRuntimeService(channel="ibm_quantum", token=os.environ["IBM_TOKEN"])
backend = service.least_busy(simulator=False, operational=True)
if use_sim:
    estimator = StatevectorEstimator()
else:
    estimator = Estimator(mode=backend)
print(f"Connected to IBM ")

spsa = qka.optimizers.SPSA(maxiter=300)
data_csv = open(data_csv_path, "a+")

generate_start_time = time.time()
problem_hamiltonian = ru.generate_ramsey_hamiltonian(graph_order, clique_order, iset_order)
generate_end_time = time.time()
generation_duration = generate_end_time - generate_start_time

compute_times = []
print(f"{"":-<80}")
for trial_i in range(num_trials):
    print(f"Executing trial {trial_i + 1}\r", end="")

    ansatz = qkl.EfficientSU2(edge_length)

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_ansatz = pm.run(ansatz)

    isa_hamiltonian = problem_hamiltonian.apply_layout(layout=isa_ansatz.layout)

    if use_sim:
        vqe = qka.VQE(estimator, ansatz, optimizer=spsa)
    else:
        vqe = qka.VQE(estimator, isa_ansatz, optimizer=spsa)
    npe = qka.NumPyMinimumEigensolver()
    initial_point = np.random.random(ansatz.num_parameters)

    compute_start_time = time.time()
    if use_sim:
        result = vqe.compute_minimum_eigenvalue(operator=problem_hamiltonian)
    else:
        result = vqe.compute_minimum_eigenvalue(operator=isa_hamiltonian)
    compute_end_time = time.time()
    compute_times.append(compute_end_time - compute_start_time)
    numpy_result = npe.compute_minimum_eigenvalue(operator=problem_hamiltonian)

    optimal_circuit = result.optimal_circuit
    optimal_params = result.optimal_parameters
    optimal_circuit.assign_parameters(optimal_params, inplace=True)

    numpy_min_energy = np.float64(numpy_result.eigenvalue)
    min_energy = result.eigenvalue
    min_energy_state = qki.Statevector(optimal_circuit).data

    state_str = ",".join([str(x) for x in min_energy_state])
    if not use_sim:
        data_csv.write(f"{graph_order};{clique_order};{iset_order};{compute_times[-1] + generation_duration};{numpy_min_energy};{min_energy};{state_str}\n")

    print(f"Trial {trial_i + 1} computation time: {compute_times[-1]} s & net time: {compute_times[-1] + generation_duration}")

data_csv.close()
print(f"\rCompleted Trials: results stored in {data_csv_path}")
