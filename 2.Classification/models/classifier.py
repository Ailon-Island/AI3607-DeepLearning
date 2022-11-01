import jittor as jt
from jittor import Module, nn


class Classifier(Module):
    def __init__(self):
        super().__init__()

    def get_optimizers(self, lr):
        optimizers = {name: jt.optim.Adam(component.parameters(), lr=lr[name] if name in lr else lr['default'])
                      for name, component in self.components.items()}
        return optimizers
