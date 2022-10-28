from jittor import nn, Module


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(1, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.loss = nn.MSELoss()

    def execute(self, x, y=None):
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x))
        x = nn.relu(self.linear3(x))
        y_pred = self.linear4(x)

        if y is None:
            return y_pred
        else:
            loss = self.loss(y_pred, y)
            return y_pred, loss
