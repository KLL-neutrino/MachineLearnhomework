import os
import time
import random
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import CNN_BN_Dropout


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def get_data_loaders_aug(batch_size=128, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=transform_train,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform_test,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_csv(history, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])

        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                history["train_loss"][i],
                history["train_acc"][i],
                history["test_loss"][i],
                history["test_acc"][i],
            ])


def save_curves(history, save_path):
    epochs = history["epoch"]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["test_loss"], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["test_acc"], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(
    save_path,
    exp_name,
    model_name,
    optimizer_name,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    dropout,
    best_acc,
    best_epoch,
    total_time,
):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Name: {exp_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Dropout: {dropout}\n")
        f.write("Data Augmentation: True\n")
        f.write("Augmentation Methods: RandomCrop(32, padding=4), RandomHorizontalFlip()\n")
        f.write(f"Best Test Accuracy: {best_acc * 100:.2f}%\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Total Time: {total_time / 60:.2f} min\n")


def main():
    set_seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    exp_name = "bn_dropout_aug_wd"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    batch_size = 128
    num_workers = 4
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 1e-4
    dropout = 0.3

    train_loader, test_loader = get_data_loaders_aug(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = CNN_BN_Dropout(num_classes=10, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc * 100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "model_name": "CNN_BN_Dropout",
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "learning_rate": learning_rate,
                    "augmentation": True,
                },
                os.path.join("checkpoints", f"{exp_name}_best.pth"),
            )

    total_time = time.time() - start_time

    log_path = os.path.join("results", f"{exp_name}_log.csv")
    curve_path = os.path.join("results", f"{exp_name}_curves.png")
    summary_path = os.path.join("results", f"{exp_name}_summary.txt")

    save_csv(history, log_path)
    save_curves(history, curve_path)

    save_summary(
        save_path=summary_path,
        exp_name=exp_name,
        model_name="CNN_BN_Dropout",
        optimizer_name="Adam",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=epochs,
        dropout=dropout,
        best_acc=best_acc,
        best_epoch=best_epoch,
        total_time=total_time,
    )

    print("=" * 60)
    print(f"Best Test Acc: {best_acc * 100:.2f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"Total Time: {total_time / 60:.2f} min")
    print(f"Saved model: checkpoints/{exp_name}_best.pth")
    print(f"Saved log: {log_path}")
    print(f"Saved curves: {curve_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
