import ramsey_util as ru
import numpy as np
import time

import qiskit as qk
from qiskit.primitives import StatevectorEstimator
import qiskit_ibm_runtime.fake_provider as qkrf
import qiskit_algorithms as qka
import qiskit.circuit.library as qkl
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2

num_experiments = 7
num_trials = 5

n_values = [4, 5, 6, 3, 4, 4, 5]
k_values = [3, 3, 3, 2, 2, 2, 2]
l_values = [3, 3, 3, 4, 4, 5, 5]
expected_energy = [0, 0, 1, 0, 1, 0, 1]
measured_energy = np.zeros((num_experiments, num_trials))
measured_duration = np.zeros((num_experiments, num_trials))

noise = False
if not noise:
    sim = AerSimulator()
else:
    backend = qkrf.FakeBrisbane()
    sim = AerSimulator.from_backend(backend)

"""
    Got the EstimatorV2 to work with the VQE problem with a custom fork and patch of the qiskit-algorithms library
"""
estimator = EstimatorV2()

print("Expected Energy Values:")
print("\t 0 - Expecting a zero energy")
print("\t 1 - Expecting a non-zero energy")

spsa = qka.optimizers.SPSA(maxiter=300)
for i in range(num_experiments):
    n = n_values[i]
    k = k_values[i]
    l = l_values[i]

    num_qubits = n * (n - 1) // 2

    for t in range(num_trials):
        start_time = time.time()
        observable = ru.generate_ramsey_hamiltonian(n, k, l)
        ansatz = qkl.EfficientSU2(num_qubits)
        ansatz = qk.transpile(ansatz, sim)
        vqe = qka.VQE(estimator, ansatz, optimizer=spsa)
        result = vqe.compute_minimum_eigenvalue(operator=observable)
        end_time = time.time()
        measured_energy[i,t] = result.eigenvalue
        measured_duration[i,t] = (end_time - start_time)

    measured_energy_mean = np.mean(measured_energy[i])
    measured_duration_mean = np.mean(measured_duration[i])
    print(f"{n_values[i]}, {k_values[i]}, {l_values[i]}, {expected_energy[i]}, {measured_energy_mean}, {measured_duration_mean}")