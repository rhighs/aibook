import sys
from scipy import optimize
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn as nn
from itertools import islice
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(1)


def mat_ops():
    t1 = 2 * torch.rand(5, 2) - 1
    t2 = torch.normal(mean=0, std=1, size=(5, 2))
    t3 = torch.multiply(t1, t2)  # same as applying t1 * t2
    print(f"pair-wise t1*t2 = {t3}")

    t4 = torch.mean(t1, axis=0)
    print(f"t1 col. mean = {t4}")

    t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
    print(f"t1 @ t2.T = {t5}")

    t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
    print(f"t1.T @ t2 = {t6}")

    norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
    print(f"t1 spectral norm = {norm_t1}")


def split_stack_concat():
    print("=== SPLITTING ===\n")
    t = torch.rand(6)
    print(f"tensor = {t}")
    t_splits = torch.chunk(t, 3)
    print(f"tensor views = {[item.numpy() for item in t_splits]}")
    print("\nwith splitting rule (explicit output sizes) = \n")
    t = torch.rand(5)
    print(f"tensor = {t}")
    t_splits = torch.split(t, split_size_or_sections=[3, 2])
    print(f"tensor views = {[item.numpy() for item in t_splits]}")

    print("\n=== STACKING ===\n")
    A = torch.ones(3)
    B = torch.zeros(3)
    S = torch.stack([A, B], axis=1)
    print(f"stack of A = {A} and B = {B} over axis 1 = {S}")


def data_loaders():
    t = torch.arange(6, dtype=torch.float32)
    print(f"simple uninitialized data loader over data = {t}")
    data_loader = DataLoader(t)
    for item in data_loader:
        print(f"load step = {item}")
    print()

    print(f"data loader with args (batch_size=3, drop_last=False) over data = {t}")
    data_loader = DataLoader(t, batch_size=3, drop_last=False)
    for item in data_loader:
        print(f"load step = {item}")
    print()

    t_x = torch.rand([4, 3], dtype=torch.float32)
    t_y = torch.arange(4)
    print(f"simple joint dataset for t_x = {t_x} and t_y = {t_y}")
    print([item for item in TensorDataset(t_x, t_y)])


def shuffle_batch_repeat():
    t_x = torch.rand([4, 3], dtype=torch.float32)
    t_y = torch.arange(4)
    joint_dataset = TensorDataset(t_x, t_y)
    data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
    print(f"shuffled batches for dataset = {[item for item in joint_dataset]}")
    for epoch in range(2):
        print(
            f"\n-----------------------------\nepoch {epoch}:\n-----------------------------\n"
        )
        for i, batch in enumerate(data_loader, 1):
            print(f"batch {i}:", "x:", batch[0], "\n\ty:", batch[1])


def torch_datasets():
    image_path = "./"
    celeba_dataset = torchvision.datasets.CelebA(
        image_path, split="train", target_type="attr", download=True
    )
    assert isinstance(celeba_dataset, torch.utils.data.Dataset)
    example = next(iter(celeba_dataset))
    print("first CelebA sample = ", example)

    fig = plt.figure(figsize=(12, 8))
    for i, (image, attributes) in islice(enumerate(celeba_dataset), 18):
        ax = fig.add_subplot(3, 6, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image)
        ax.set_title(f"{attributes[31]}", size=15)
    plt.show()

    mnist_dataset = torchvision.datasets.MNIST(image_path, "train", download=True)
    assert isinstance(mnist_dataset, torch.utils.data.Dataset)
    example = next(iter(mnist_dataset))
    print("first MNIST sample = ", example)
    fig = plt.figure(figsize=(12, 8))
    for i, (image, label) in islice(enumerate(mnist_dataset), 10):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image, cmap='gray_r')
        ax.set_title(f"{label}", size=15)
    plt.show()

from torch.utils.data import TensorDataset
def linear_regression():
    X_train = np.arange(10, dtype='float32').reshape((10, 1))
    y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')
    plt.plot(X_train, y_train, 'o', markersize=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    X_train_norm = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm)
    y_train = torch.from_numpy(y_train).float()
    train_ds = TensorDataset(X_train_norm, y_train)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    weight = torch.randn(1)
    weight.requires_grad_()
    bias = torch.zeros(1, requires_grad=True)

    def model(xb):
        return xb @ weight + bias

    def loss_fn(input, target):
        return (input - target).pow(2).mean()

    learning_rate = 0.001
    num_epochs = 200
    log_epochs = 10

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch.long())

            # calculate and apply gradient, storing them in weight.grad and bias.grad
            # this thing holds references to the computational graph,
            # hence the need of setting requires_grad=True some lines above...
            loss.backward()
        with torch.no_grad():
            weight -= weight.grad * learning_rate
            bias -= bias.grad * learning_rate
            weight.grad.zero_()
            bias.grad.zero_()
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch} - Loss {loss.item():.4f} (loss raw value: {loss})")
    print(f"Final params: {weight.item()} {bias.item()}")

    X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
    X_test_norm = (X_test - np.mean(X_test)) / np.std(X_test)
    X_test_norm = torch.from_numpy(X_test_norm)

    # detach() creates a new tensor detached from the current computation graph
    y_pred = model(X_test_norm).detach().numpy()
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(X_train_norm, y_train, 'o', markersize= 10)
    plt.plot(X_test_norm, y_pred, '--', lw=3)
    plt.legend(['Training examples', 'Linear reg.'], fontsize=15)
    ax.set_xlabel('x', size=15)
    ax.set_ylabel('y', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

def training_via_torch_nn():
    X_train = np.arange(10, dtype='float32').reshape((10, 1))
    y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

    X_train_norm = (X_train - (np.mean(X_train))) / np.std(X_train)
    X_train_norm = torch.from_numpy(X_train_norm)
    y_train = torch.from_numpy(y_train).float()
    train_ds = TensorDataset(X_train_norm, y_train)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    learning_rate = 0.001
    num_epochs = 200
    log_epochs = 10

    loss_fn = nn.MSELoss(reduction='mean')
    input_size = 1
    output_size = 1
    model = nn.Linear(input_size, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            # compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()
            # reset gradients values
            optimizer.zero_grad()
        if epoch % log_epochs == 0:
            print(f"Epoch {epoch} - Loss {loss.item():.4f} (loss raw value: {loss})")
    print('Final Parameters:', model.weight.item(), model.bias.item())

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        return x

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
def iris_classifier_multilayer():
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1./3, random_state=1
    )

    X_train_norm = torch.from_numpy((X_train - np.mean(X_train)) / np.std(X_train)).float()
    y_train = torch.from_numpy(y_train)
    train_ds = TensorDataset(X_train_norm, y_train)
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=True)

    input_size = X_train_norm.shape[1]
    hidden_size = 16
    output_size = 3
    model = Model(input_size, hidden_size, output_size)

    learning_rate = 0.001
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 100
    loss_hist = [0] * num_epochs
    accuracy_hist = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)

            # casting to long here fixes: RuntimeError: expected scalar type Long but found Int 
            loss = loss_fn(pred, y_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist[epoch] += loss.item()*y_batch.size(0)

            # get index for the predicted output and check if it matches with the expected trainng label
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.sum()
        loss_hist[epoch] /= len(train_dl.dataset)
        accuracy_hist[epoch] /= len(train_dl.dataset)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(loss_hist, lw=3)
    ax.set_title('Training loss', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(accuracy_hist, lw=3)
    ax.set_title('Training accuracy', size=15)
    ax.set_xlabel('Epoch', size=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.show()

    X_test_norm = torch.from_numpy((X_test - np.mean(X_test)) / np.std(X_test)).float()
    y_test = torch.from_numpy(y_test)
    pred_test = model(X_test_norm)
    correct = (torch.argmax(pred_test, dim=1) == y_test).float()
    accuracy = correct.mean()
    print(f"[in-memory-model]: Test dataset accuracy: {accuracy:.4f}")

    model_path = "iris_classifier.pt"
    # saves the full net architecture
    torch.save(model, model_path)
    #
    # to only save params:
    # torch.save(model.state_dict(), model_path)
    # then loading it with: 
    #   model_new = Model(input_size, hidden_size, output_size)
    #   model_new.load_state_dict(torch.load(model_path))
    #
    model_new = torch.load(model_path)
    print("Saved and loaded model eval: ", model_new.eval())
    pred_test = model_new(X_test_norm)
    correct = (torch.argmax(pred_test, dim=1) == y_test).float()
    accuracy = correct.mean()
    print(f"[loaded-model]: Test dataset accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    globals()[sys.argv[1]]()
