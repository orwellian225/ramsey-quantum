import matplotlib.pyplot as plt
import qiskit as qk
import qiskit.circuit as qkc
import ramsey_util as ru

n = 5
l = n * (n - 1) // 2
circuit = qk.QuantumCircuit(l)
ham = ru.generate_ramsey_hamiltonian(n, 3, 3)
circuit.append(
    qkc.library.PauliEvolutionGate(ham, time=0.1, label='a'),
    qargs=[x for x in range(l)]
)

circuit.draw("mpl")
plt.show()
circuit = circuit.decompose()
circuit.draw("mpl")
plt.show()
print(circuit.count_ops())