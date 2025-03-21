from lazyqml.Interfaces.iCircuit import Circuit
import pennylane as qml

class RyEmbedding(Circuit):
    def __init__(self) -> None:
        super().__init__()

    def getCircuit(self):
        def ry_embedding(x, wires):
            """Embeds a quantum state into the quantum device using rotation around the Y-axis.

            Args:
                x (array[float]): array of rotation angles for each qubit
                wires (Sequence[int]): wires that the operation acts on

            Returns:
                None
            """
            qml.AngleEmbedding(x, wires=wires, rotation='Y')
        return ry_embedding