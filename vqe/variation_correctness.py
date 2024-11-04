import ramsey_util as ru
import numpy as np

import qiskit as qk
import qiskit_ibm_runtime.fake_provider as qkrf
import qiskit_algorithms as qka
import qiskit.circuit.library as qkl
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler

num_experiments = 7
num_trials = 1

n_values = [4, 5, 6, 3, 4, 4, 5]
k_values = [3, 3, 3, 2, 2, 2, 2]
l_values = [3, 3, 3, 4, 4, 5, 5]
expected_energy = [0, 0, 1, 0, 1, 0, 1]
measured_energy = np.zeros((num_experiments, num_trials))

noise = False

if not noise:
    sim = AerSimulator()
else:
    backend = qkrf.FakeBrisbane()
    sim = AerSimulator.from_backend(backend)
sampler = Sampler(run_options={'shots': 1000})

print("Expected Energy Values:")
print("\t 0 - Expecting a zero energy")
print("\t 1 - Expecting a non-zero energy")

spsa = qka.optimizers.SPSA(maxiter=300)
for i in range(num_experiments):
    n = n_values[i]
    k = k_values[i]
    l = l_values[i]

    num_qubits = n * (n - 1) // 2

    observable = ru.generate_ramsey_hamiltonian(n, k, l)
    for t in range(num_trials):
        ansatz = qkl.EfficientSU2(num_qubits)
        ansatz = qk.transpile(ansatz, sim)
        vqe = qka.SamplingVQE(sampler, ansatz, optimizer=spsa)
        result = vqe.compute_minimum_eigenvalue(operator=observable)
        measured_energy[i,t] = result.eigenvalue

    # print(f"n, k, l = {n}, {k}, {l}")
    # print(f"\tAvailable states: {[x for x in range(2**num_qubits)]}")
    # print(f"\tEnergy: {result.eigenvalue}")
    # print(f"\tState: {result.eigenstate}")

    measured_energy_mean = np.mean(measured_energy[i])
    print(f"{n_values[i]}, {k_values[i]}, {l_values[i]}, {expected_energy[i]}, {measured_energy_mean}")