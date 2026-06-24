import csv
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    return device


def accuracy_from_logits(outputs, labels):
    _, predicted = outputs.max(1)
    correct = predicted.eq(labels).sum().item()
    return correct, labels.size(0)


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
        batch_correct, batch_total = accuracy_from_logits(outputs, labels)
        correct += batch_correct
        total += batch_total

    return running_loss / total, correct / total


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
        batch_correct, batch_total = accuracy_from_logits(outputs, labels)
        correct += batch_correct
        total += batch_total

    return running_loss / total, correct / total


def save_csv(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['test_loss'][i]:.6f}",
                f"{history['test_acc'][i]:.6f}",
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
    plt.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="Train Acc")
    plt.plot(epochs, [acc * 100 for acc in history["test_acc"]], label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary(save_path, args, best_acc, best_epoch, total_time):
    lines = [
        f"Experiment name: {args.exp_name}",
        f"Model: {args.model}",
        f"Data augmentation: {args.augment}",
        f"Optimizer: {args.optimizer}",
        f"Learning rate: {args.lr}",
        f"Weight decay: {args.weight_decay}",
        f"Batch size: {args.batch_size}",
        f"Epochs: {args.epochs}",
        f"Best test accuracy: {best_acc * 100:.2f}%",
        f"Best epoch: {best_epoch}",
        f"Total training time: {total_time / 60:.2f} min",
    ]
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def elapsed_minutes(start_time):
    return (time.time() - start_time) / 60
