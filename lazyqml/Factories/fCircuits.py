# Importing Enums
from lazyqml.Global.globalEnums import Ansatzs, Embedding
# Importing Circuits
from lazyqml.Circuits.Ansatzs import HardwareEfficient, HCzRx, TreeTensor, TwoLocal, Annular
from lazyqml.Circuits.Embeddings import DenseAngleEmbedding, ZZEmbedding, HigherOrderEmbedding

from functools import partial
import pennylane as qml

class CircuitFactory:
    def __init__(self, nqubits, nlayers) -> None:
        self.nqubits = nqubits 
        self.nlayers = nlayers

    def GetAnsatzCircuit(self,ansatz):
        if ansatz == Ansatzs.HARDWARE_EFFICIENT:
            return HardwareEfficient(self.nqubits, self.nlayers)
        elif ansatz == Ansatzs.HCZRX:
            return HCzRx(self.nqubits, self.nlayers)
        elif ansatz == Ansatzs.TREE_TENSOR:
            return TreeTensor(self.nqubits, nlayers=self.nlayers)
        elif ansatz == Ansatzs.TWO_LOCAL:
            return TwoLocal(self.nqubits, nlayers=self.nlayers)
        elif ansatz == Ansatzs.ANNULAR:
            return Annular(self.nqubits, nlayers=self.nlayers)

    def GetEmbeddingCircuit(self, embedding):
        if embedding == Embedding.RX:
            return partial(qml.AngleEmbedding, rotation='X')
        
        elif embedding == Embedding.RY:
            return partial(qml.AngleEmbedding, rotation='Y')
        
        elif embedding == Embedding.RZ:
            def RZ(inputs, wires):
                [qml.Hadamard(i) for i in wires]
                qml.AngleEmbedding(inputs, wires, rotation='Z')

            return RZ
        
        elif embedding == Embedding.ZZ:
            return ZZEmbedding
        
        elif embedding == Embedding.AMP:
            return partial(qml.AmplitudeEmbedding, pad_with=0, normalize=True)
        
        elif embedding == Embedding.DENSE_ANGLE:
            return DenseAngleEmbedding
        
        elif embedding == Embedding.HIGHER_ORDER:
            return HigherOrderEmbedding