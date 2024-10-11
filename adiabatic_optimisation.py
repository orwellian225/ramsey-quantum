import qiskit as qk
import qiskit.circuit as qkc
import qiskit.quantum_info as qki

class AdiabaticOptimisationProblem:
    def __init__(
            self,
            initial_hamiltonian: qki.operators.Operator,
            problem_hamiltonian: qki.operators.Operator,
            time: float,
            steps: int
        ):

        if initial_hamiltonian.num_qubits != problem_hamiltonian.num_qubits:
            raise ValueError(f"Initial Hamiltonian and Problem Hamiltonian have incorrect dimensions, {initial_hamiltonian.num_qubits} != {problem_hamiltonian.num_qubits}")

        self.initial_hamiltonian = initial_hamiltonian
        self.problem_hamiltonian = problem_hamiltonian
        self.num_qubits = self.initial_hamiltonian.num_qubits
        self.qubit_list = [x for x in range(self.num_qubits)]

        self.time = time
        self.steps = steps

        self.delta = self.time / self.steps
        self.circuit = qk.QuantumCircuit(self.num_qubits)

    def generate(self, initial_state):
        self.circuit.initialize(initial_state)
        for step in range(self.steps):
            interpolated_op = (step * self.delta / self.time) * self.initial_hamiltonian + (1 - step * self.delta / self.time) * self.problem_hamiltonian
            local = qkc.library.PauliEvolutionGate(interpolated_op, time=self.delta, label=f"U({(step + 1) * self.delta},{step * self.delta})")
            self.circuit.append(local, qargs=self.qubit_list)
