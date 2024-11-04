from qiskit.quantum_info.operators import SparsePauliOp, Pauli
from util import enumerate_choices, pauli_at

def generate_initial_hamiltonian(graph_order: int) -> SparsePauliOp:
    length = graph_order * (graph_order - 1) // 2
    result = SparsePauliOp([pauli_at("I", 0, length)], [0.])
    for l in range(length):
        result += SparsePauliOp([pauli_at("I", l, length), pauli_at("X", l, length)], [0.5, -0.5])

    return result

def generate_clique_hamiltonian(graph_order: int, clique_size: int) -> SparsePauliOp:
    length = graph_order * (graph_order - 1) // 2
    vertex_choices = enumerate_choices(graph_order, clique_size)
    idx = lambda n,r,c: int(-0.5 * r**2 + (n - 0.5) * r + c - r)

    result = SparsePauliOp([pauli_at("I", 0, length)], [0.])
    for vc in vertex_choices:
        h_alpha = Pauli(pauli_at("I", 0, length))
        edge_choices = enumerate_choices(clique_size, 2)

        for ec in edge_choices:
            edge_idx = idx(graph_order, vc[ec[0]], vc[ec[1]])
            edge_pauli = SparsePauliOp([pauli_at("I", edge_idx, length), pauli_at("Z", edge_idx, length)], [0.5, -0.5])
            h_alpha = edge_pauli.dot(h_alpha)

        result += h_alpha

    return result

def generate_iset_hamiltonian(graph_order: int, iset_size: int) -> SparsePauliOp:
    length = graph_order * (graph_order - 1) // 2
    vertex_choices = enumerate_choices(graph_order, iset_size)
    idx = lambda n,r,c: int(-0.5 * r**2 + (n - 0.5) * r + c - r)

    result = SparsePauliOp([pauli_at("I", 0, length)], [0.])
    for vc in vertex_choices:
        h_alpha = Pauli(pauli_at("I", 0, length))
        edge_choices = enumerate_choices(iset_size, 2)

        for ec in edge_choices:
            edge_idx = idx(graph_order, vc[ec[0]], vc[ec[1]])
            edge_pauli = SparsePauliOp([pauli_at("I", edge_idx, length), pauli_at("Z", edge_idx, length)], [0.5, 0.5])
            h_alpha = edge_pauli.dot(h_alpha)

        result += h_alpha

    return result

def generate_ramsey_hamiltonian(graph_order: int, clique_size: int, iset_size: int) -> SparsePauliOp:
    clique_ham = generate_clique_hamiltonian(graph_order, clique_size)
    iset_ham = generate_iset_hamiltonian(graph_order, iset_size)

    return clique_ham + iset_ham
