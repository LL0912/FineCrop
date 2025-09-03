from abc import ABC, abstractmethod
from SITS.models.build_model import build_model
import torch
class BaseInfer(ABC):
    def __init__(self,model,meta):
        self.cfg_model = model
        self.meta = meta
        self.model=build_model(self.cfg_model)
        self._build_device()


    def _build_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def infer_image(self):
        pass
