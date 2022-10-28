import jittor as jt
from jittor import Module, nn


class Aggregator(Module):
    def __init__(self, in_channel=512, num_pos=4):
        super(Aggregator, self).__init__()
        self.linear1 = nn.Linear(in_channel * num_pos, 4096)
        self.linear2 = nn.Linear(4096, num_pos ** 2)

    def execute(self, x):
        x = nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x
