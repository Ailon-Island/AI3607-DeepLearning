import os
from matplotlib import pyplot as plt

import numpy as np
import jittor as jt
from jittor import distributions

from model import Net


def get_data(mean=0, std=1, interval=[-5, 5], num_samples=1000, split=0.8):
    gaussian = distributions.Normal(mean, std)
    pdf = lambda xx: gaussian.log_prob(xx).exp()

    x = jt.rand(num_samples) * (interval[1] - interval[0]) + interval[0]
    y = pdf(x)

    # Split the data
    train_data = (x[:int(num_samples*split)], y[:int(num_samples*split)])
    test_data = (x[int(num_samples*split):], y[int(num_samples*split):])

    return train_data, test_data, pdf


def train(epoch_idx, model, optimizer, train_data):
    model.train()

    x, y = train_data
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    y_pred, loss = model(x, y)
    train_loss = loss.item()
    optimizer.step(loss)

    print(f'Epoch: {epoch_idx} \tTraining Loss: {train_loss:.6f}')

    return train_loss


def test(model, test_data):
    model.eval()

    with jt.no_grad():
        x, y = test_data
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        y_pred, loss = model(x, y)
        test_loss = loss.item()

    print(f'\t\t\tTest Loss: {test_loss:.6f}')

    return test_loss


def plot(train_losses, test_losses, pdf, model, interval=[-5, 5], num_samples=1000):
    if not os.path.exists('results'):
        os.mkdir('results')

    num_epoch = len(train_losses)
    epochs = np.arange(1, num_epoch+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join('results', 'loss.png'))

    plt.figure(figsize=(10, 5))
    x = np.linspace(*interval, num_samples)
    x_in = jt.array(x).reshape(-1, 1)
    y = pdf(x_in)
    y_pred = model(x_in)
    y, y_pred = y.data, y_pred.data
    plt.plot(x, y, label='True', linewidth=6)
    plt.plot(x, y_pred, label='Predicted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join('results', 'pdf.png'))


def main():
    # Generate some data
    train_data, test_data, pdf = get_data(**data_kwargs)

    # Load the model
    model = Net()
    optimizer = jt.optim.Adam(model.parameters(), lr=lr)

    # Train & test the model
    train_losses, test_losses = [], []
    for epoch_idx in range(1, num_epoch+1):
        train_loss = train(epoch_idx, model, optimizer, train_data)
        test_loss = test(model, test_data)

        train_losses += [train_loss]
        test_losses += [test_loss]

    # Plot the results
    plot(train_losses, test_losses, pdf, model, interval=data_kwargs['interval'], num_samples=data_kwargs['num_samples'])







if __name__ == "__main__":
    jt.flags.use_cuda = 1
    jt.set_global_seed(0)

    data_kwargs = {
        'mean': 0,
        'std': 1,
        'interval': [-5, 5],
        'num_samples': 1000,
        'split': 0.8
    }
    num_epoch = 1000
    lr = 3e-4

    main()


