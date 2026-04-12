from dataclasses import dataclass
from typing import Any, Optional, Dict

from dataclasses import dataclass
from typing import Any, Optional, Dict

@dataclass
class QMLTask:
    # --- identifiers --- 
    id: int

    # --- model ---
    model: Any
    model_memory: float
    nqubits: int
    model_params: Optional[Dict] = None

    # --- data ---
    X_train: Any = None
    X_test:  Any = None
    y_train: Any = None
    y_test:  Any = None

    # --- evaluation ---
    custom_metric: Any = None

    def get_model_params(self):
        return self.model_params

    def get_data(self):
        return (self.X_train, self.y_train, self.X_test, self.y_test)

    
    def get_task_params(self):
        # print((self.model_params, *self.get_data(), self.custom_metric))
        return (self.id, self.model_params, *self.get_data(), self.custom_metric)