import sys
import os

from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

import qiskit as qk
import qiskit_algorithms as qka
import qiskit.quantum_info as qki
import qiskit.circuit.library as qkl
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

service = QiskitRuntimeService(channel="ibm_quantum", token=os.environ["IBM_TOKEN"])
backend = service.least_busy(simulator=False, operational=True)
estimator = Estimator(mode=backend)
spsa = qka.optimizers.SPSA(maxiter=300)
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

ansatz = qkl.EfficientSU2(2)
H2_op = qki.operators.SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

isa_ansatz = pm.run(ansatz)
isa_op = H2_op.apply_layout(layout=isa_ansatz.layout)

vqe = qka.VQE(estimator, isa_ansatz, optimizer=spsa)
result = vqe.compute_minimum_eigenvalue(operator=isa_op)

print(result.eigenvalue)
