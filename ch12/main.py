import sys
import torch
import matplotlib.pyplot as plt
from itertools import islice
import torchvision
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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
