# Importing Enums
from lazyqml.Global.globalEnums import Ansatzs, Embedding
# Importing Circuits
from lazyqml.Circuits import *

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
            return RxEmbedding()
        elif embedding == Embedding.RY:
            return RyEmbedding()
        elif embedding == Embedding.RZ:
            return RzEmbedding()
        elif embedding == Embedding.ZZ:
            return ZzEmbedding()
        elif embedding == Embedding.AMP:
            return AmplitudeEmbedding()
        elif embedding == Embedding.DENSE_ANGLE:
            return DenseAngleEmbedding()
        elif embedding == Embedding.HIGHER_ORDER:
            return HigherOrderEmbedding()