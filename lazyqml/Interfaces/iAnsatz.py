from abc import abstractmethod
from lazyqml.Interfaces.iCircuit import Circuit

class Ansatz(Circuit):
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers

        self._n_ansatz_params = None

    @property
    @abstractmethod
    def n_ansatz_params(self):
        pass
    
    @property
    def n_total_params(self):
        return self.n_ansatz_params * self.nlayers