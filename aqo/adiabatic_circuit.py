import matplotlib as mpl
import matplotlib.pyplot as plt
import ramsey_util as ru
import seaborn as sns
import polars as pl
import numpy as np
import math as m

from adiabatic_optimisation import AdiabaticOptimisationProblem
import qiskit as qk
import qiskit_algorithms as qka
from qiskit import transpile

k = 3
l = 3
n = 3

initial_ham = ru.generate_initial_hamiltonian(n)
problem_ham = ru.generate_ramsey_hamiltonian(n, k, l)

aqo = AdiabaticOptimisationProblem(
    initial_ham, problem_ham, 3., 2
)

aqo.generate([1 / m.sqrt(2**aqo.num_qubits) for _ in range(2**aqo.num_qubits)])

# aqo.circuit = transpile(aqo.circuit, optimization_level=3)
aqo.circuit.draw("mpl", filename=f"./visualizations/circuit_AQO.pdf")
aqo.circuit = aqo.circuit.decompose()
print(aqo.circuit.depth())

# aqo.circuit.draw("mpl", filename=f"./visualizations/circuit_AQO.pdf", style={
#     "backgroundcolor": '#1b1b1b',
#     "textcolor": "white",
#     "linecolor": "white"
# })


aqo.circuit.draw("mpl", filename=f"./visualizations/raw_circuit_AQO.pdf")