import ramsey_util as ru
import numpy as np
import time

import qiskit as qk
import qiskit_ibm_runtime.fake_provider as qkrf
import qiskit_algorithms as qka
import qiskit.circuit.library as qkl
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2

num_experiments = 5 * 3 

n_values = [i for i in range(5, 11)] * 3
k_values = [3] * 5 + [4] * 5 + [5] * 5
l_values = [3] * 5 + [4] * 5 + [5] * 5

noise = False
if not noise:
    sim = AerSimulator()
else:
    backend = qkrf.FakeBrisbane()
    sim = AerSimulator.from_backend(backend)
sampler = SamplerV2(run_options={'shots': 1000})


spsa = qka.optimizers.SPSA(maxiter=300)
for i in range(num_experiments):
    n = n_values[i]
    k = k_values[i]
    l = l_values[i]

    num_qubits = n * (n - 1) // 2

    start_time = time.time()
    observable = ru.generate_ramsey_hamiltonian(n, k, l)
    ansatz = qkl.EfficientSU2(num_qubits)
    ansatz = qk.transpile(ansatz, sim)
    vqe = qka.SamplingVQE(sampler, ansatz, optimizer=spsa)
    result = vqe.compute_minimum_eigenvalue(operator=observable)
    end_time = time.time()

    print(f"{n_values[i]},{k_values[i]},{l_values[i]},{end_time - start_time}")
    if noise:
        f = open('raw_data/vqe_noisy_long.csv', 'a+')
    else:
        f = open('raw_data/vqe_long.csv', 'a+')
    f.write(f"{n_values[i]},{k_values[i]},{l_values[i]}, {end_time - start_time},{result.eigenvalue},{result.eigenstate}\n")
    f.close()
