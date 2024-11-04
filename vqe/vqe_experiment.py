import ramsey_util as ru
import numpy as np

from qiskit_aer import AerSimulator, StatevectorSimulator 
from qiskit_aer.primitives import EstimatorV2, SamplerV2, Estimator
import qiskit_algorithms as qka
import qiskit as qk
import qiskit.quantum_info as qki
import qiskit.circuit.library as qkl

n = 3
length = n * (n - 1) // 2
print(f"Problem: R(3,3) ?= {n}")
print(f"Number of qubits: {length}")
problem_hamiltonian = ru.generate_ramsey_hamiltonian(n, 3, 3)
print("Numpy Min Energy:", np.min(np.linalg.eigvals(problem_hamiltonian)))

simulator = StatevectorSimulator(device="GPU")
estimator = Estimator()

ansatz = qkl.EfficientSU2(length)
initial_point = np.random.random(ansatz.num_parameters)
spsa = qka.optimizers.SPSA(maxiter=300)

decomposed_ansatz = ansatz.decompose()
decomposed_ansatz.draw("mpl", filename=f"./visualizations/raw_circuit_VQE.pdf")

ansatz.draw("mpl", filename=f"./visualizations/circuit_VQE.pdf", style={
    "backgroundcolor": '#1b1b1b',
    "textcolor": "white",
    "linecolor": "white"
})

vqe = qka.VQE(estimator, ansatz, optimizer=spsa)
result = vqe.compute_minimum_eigenvalue(operator=problem_hamiltonian)

optimal_circuit = result.optimal_circuit
optimal_params = result.optimal_parameters

optimal_circuit.assign_parameters(optimal_params, inplace=True)
optimal_state = qki.Statevector(optimal_circuit)

print("VQE Min Energy:", result.eigenvalue)
print(f"Likely VQE Min Energy state: |{bin(np.argmax(optimal_state.data))[2:]}> with probability {np.max(optimal_state.data)**2}")
print("VQE Min Energy state probabilites:")
for i, x in enumerate(optimal_state.data):
    print(f"\tP(|{bin(i)[2:]}>) = {x.real**2 + x.imag**2}")

