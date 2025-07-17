from lazyqml.Interfaces.iAnsatz import Ansatz
import pennylane as qml

class Annular(Ansatz):
    def getCircuit(self):
        def annular(theta, wires):
            """Implements an annular ansatz circuit.

            Args:
                theta (array[float]): array of parameters for the ansatz circuit
                wires (Sequence[int]): wires that the ansatz circuit acts on

            Returns:
                None
            """

            N=len(wires)

            param_count = 0

            for _ in range(self.nlayers):
                for i in range(N):
                    qml.X(wires=i)
                    qml.Hadamard(wires=i)

                for i in range(N - 1):
                    qml.CNOT(wires = [i, i + 1])
                    qml.RY(theta[param_count], wires = i+1)

                    param_count += 1

                qml.CNOT(wires=[N-1, 0])
                qml.RY(theta[param_count], wires = 0)
                param_count += 1

        return annular
    
    @property
    def n_ansatz_params(self):
        return self.nqubits