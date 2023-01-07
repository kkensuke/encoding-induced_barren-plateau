import numpy as np

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi


def analytical(s,n):
    return (2**s - 2**(-s))/(2**n + 1)


# HSD : Hilbert Schmidt distance
# HEE : Hardware Efficient Embedding
# MMS : Maximally Mixed State

class HSD:
    def __init__(self, nqubits, nlayers, s, repeat_params = True):
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.s = s
        self.repeat_params = repeat_params

    def HEE(self, circuit, params):
        for i in range(self.nqubits):
            circuit.rx(params[i], i)
            circuit.ry(params[i + self.nqubits], i)
        for i in range(self.nqubits - 1):
            circuit.cx(i, i+1)

    def HSD_from_MMS(self, params):
        circuit = QuantumCircuit(self.nqubits)

        if params is None:
            if self.repeat_params:
                params = np.random.uniform(0, 2*np.pi, 2*self.nqubits)
                for i in range(self.nlayers):
                    self.HEE(circuit, params)
                    circuit.barrier()
            else:
                params = np.random.uniform(0, 2*np.pi, 2*self.nqubits*self.nlayers)
                for i in range(self.nlayers):
                    self.HEE(circuit, params[2*self.nqubits*i:2*self.nqubits*(i+1)])
                    circuit.barrier()
        else:
            if self.repeat_params:
                for i in range(self.nlayers):
                    self.HEE(circuit, params)
                    circuit.barrier()
            else:
                for i in range(self.nlayers):
                    self.HEE(circuit, params[2*self.nqubits*i:2*self.nqubits*(i+1)])
                    circuit.barrier()

        rho = qi.DensityMatrix.from_instruction(circuit)
        partial_rho = qi.partial_trace(rho, list(np.arange(self.nqubits - self.s)))

        return qi.purity(partial_rho) - 1/2**self.s