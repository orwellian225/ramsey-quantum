import os
import sys
import ramsey_util as ru
import numpy as np

import qiskit as qk
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
import qiskit_algorithms as qka
import qiskit.circuit.library as qkl
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

num_experiments = 7
num_trials = 1

n_values = [4, 5, 6, 3, 4, 4, 5]
k_values = [3, 3, 3, 2, 2, 2, 2]
l_values = [3, 3, 3, 4, 4, 5, 5]
expected_energy = [0, 0, 1, 0, 1, 0, 1]
measured_energy = np.zeros((num_experiments, num_trials))

service = QiskitRuntimeService(channel="ibm_quantum", token=os.environ["IBM_TOKEN"])
backend = service.least_busy(simulator=False, operational=True)
sampler = SamplerV2()

print("Expected Energy Values:")
print("\t 0 - Expecting a zero energy")
print("\t 1 - Expecting a non-zero energy")

spsa = qka.optimizers.SPSA(maxiter=300)
# for i in range(num_experiments):

if len(sys.argv) != 2:
    i = 0
else:
    i = int(sys.argv[1])
    if 0 <= i and i < 7:
        print("Invalid experiment")
        exit()

n = n_values[i]
k = k_values[i]
l = l_values[i]

num_qubits = n * (n - 1) // 2

ansatz = qkl.EfficientSU2(num_qubits)
observable = ru.generate_ramsey_hamiltonian(n, k, l)

pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
isa_ansatz = pm.run(ansatz)
isa_hamiltonian = observable.apply_layout(layout=isa_ansatz.layout)

vqe = qka.SamplingVQE(sampler, isa_ansatz, optimizer=spsa)
result = vqe.compute_minimum_eigenvalue(operator=isa_hamiltonian)
measured_energy = result.eigenvalue
print(f"{n_values[i]}, {k_values[i]}, {l_values[i]}, {expected_energy[i]}, {measured_energy}")