import os
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import CNN_BN_Dropout


classes = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


def get_test_loader(batch_size=128, num_workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=transform_test,
    )

    pin_memory = torch.cuda.is_available()

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


@torch.no_grad()
def compute_confusion_matrix(model, test_loader, device, num_classes=10):
    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    total = 0
    correct = 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, preds = outputs.max(1)

        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        for true_label, pred_label in zip(labels.cpu(), preds.cpu()):
            confusion[true_label, pred_label] += 1

    acc = correct / total

    return confusion.numpy(), acc


def save_confusion_csv(confusion, save_path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + list(classes))

        for i, row in enumerate(confusion):
            writer.writerow([classes[i]] + row.tolist())


def save_class_accuracy(confusion, save_path):
    class_acc = confusion.diagonal() / confusion.sum(axis=1)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])

        for i, acc in enumerate(class_acc):
            writer.writerow([
                classes[i],
                int(confusion[i, i]),
                int(confusion[i].sum()),
                float(acc),
            ])

    return class_acc


def plot_confusion_matrix(confusion, save_path):
    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(confusion)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix on CIFAR-10 Test Set")

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                str(confusion[i, j]),
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def print_top_confusions(confusion, top_k=10):
    items = []

    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            if i != j:
                items.append((confusion[i, j], classes[i], classes[j]))

    items.sort(reverse=True, key=lambda x: x[0])

    print("\nTop Confusions:")
    for count, true_class, pred_class in items[:top_k]:
        print(f"{true_class:>10s} -> {pred_class:<10s}: {count}")


def main():
    os.makedirs("results", exist_ok=True)

    ckpt_path = "checkpoints/bn_dropout_aug_wd_sgd_momentum_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_loader = get_test_loader(batch_size=128, num_workers=4)

    model = CNN_BN_Dropout(num_classes=10, dropout=0.3).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint best epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint best acc: {checkpoint.get('best_acc', 0.0) * 100:.2f}%")

    confusion, test_acc = compute_confusion_matrix(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=10,
    )

    print(f"\nTest Accuracy from loaded model: {test_acc * 100:.2f}%")

    confusion_csv_path = "results/sgd_momentum_confusion_matrix.csv"
    class_acc_csv_path = "results/sgd_momentum_class_accuracy.csv"
    confusion_png_path = "results/sgd_momentum_confusion_matrix.png"

    save_confusion_csv(confusion, confusion_csv_path)
    class_acc = save_class_accuracy(confusion, class_acc_csv_path)
    plot_confusion_matrix(confusion, confusion_png_path)

    print("\nPer-class Accuracy:")
    for cls, acc in zip(classes, class_acc):
        print(f"{cls:>10s}: {acc * 100:.2f}%")

    print_top_confusions(confusion, top_k=10)

    print("\nSaved files:")
    print(f"Confusion matrix CSV: {confusion_csv_path}")
    print(f"Class accuracy CSV: {class_acc_csv_path}")
    print(f"Confusion matrix figure: {confusion_png_path}")


if __name__ == "__main__":
    main()
