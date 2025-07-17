# Importing from
from lazyqml.Global.globalEnums import *
from lazyqml.Preprocessing import PCAHelper, Sanitizer

class PreprocessingFactory:
    def __init__(self, nqubits) -> None:
        self.nqubits = nqubits

    def GetSanitizer(self, imputerCat, imputerNum):
        return Sanitizer(imputerCat, imputerNum)

    def GetPreprocessing(self, embedding, ansatz):
        if embedding == Embedding.AMP and ansatz == Ansatzs.TREE_TENSOR:
            return PCAHelper(self.nqubits, 2**(2**(self.nqubits.bit_length()-1)))
        elif embedding == Embedding.AMP:
            return PCAHelper(self.nqubits, 2**self.nqubits)
        elif embedding == Embedding.DENSE_ANGLE:
            return PCAHelper(self.nqubits, 2*self.nqubits)
        elif ansatz == Ansatzs.TREE_TENSOR:
            return PCAHelper(self.nqubits, 2**(self.nqubits.bit_length()-1))
        else:
            return PCAHelper(self.nqubits, self.nqubits)