from qiskit.quantum_info.operators import SparsePauliOp
import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import numpy as np

def visualize_hamiltonian(hamiltonian: np.ndarray, name: str):
    sns.set_style("whitegrid")
    ax = sns.heatmap(data=hamiltonian, cmap="inferno", cbar=False)
    ax.set_aspect("equal")
    _ = ax.set_title(f"{name} Hamiltonian")
    _ = ax.axis('off')
    
    plt.tight_layout()
    filename = name.lower().replace(" ", "_")
    plt.savefig(f"./visualizations/{filename}_hamiltonian_heatmap.png")

def main():
    ham_n3_k2: SparsePauliOp = ru.generate_clique_hamiltonian(3, 2)
    visualize_hamiltonian(ham_n3_k2.to_matrix().astype(np.float32), "Order 3 Clique 2")

    ham_n3_l2: SparsePauliOp = ru.generate_iset_hamiltonian(3, 2)
    visualize_hamiltonian(ham_n3_l2.to_matrix().astype(np.float32), "Order 3 ISet 2")

    # ham_n5_k3: SparsePauliOp = ru.generate_clique_hamiltonian(5, 3)
    # visualize_hamiltonian(ham_n5_k3.to_matrix().astype(np.float32), "Order 5 Clique 3")

if __name__ == "__main__":
    main()
