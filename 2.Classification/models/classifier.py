import jittor as jt
from jittor import Module, nn


class Classifier(Module):
    def __init__(self):
        super().__init__()

    def get_optimizer(self, lr, weight_decay):
        param_groups = [{'params': [], 'lr': lr['default']}]
        for name, module in self.components.items():
            if name in lr:
                param_groups.append({'params': module.parameters(), 'lr': lr[name]})
            else:
                param_groups[0]['params'].extend(module.parameters())
        optimizer = jt.optim.Adam(param_groups, lr=lr['default'], weight_decay=weight_decay)
        return optimizer
