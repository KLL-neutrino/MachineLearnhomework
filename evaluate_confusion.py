import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import CIFAR10_CLASSES, ensure_project_dirs, get_data_loaders
from model import build_model
from utils import get_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Save a CIFAR-10 confusion matrix.")
    parser.add_argument("--model", choices=["baseline", "cnn_bn_dropout"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(labels.cpu())

    return torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy()


def compute_confusion_matrix(labels, preds, num_classes=10):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, preds):
        matrix[label, pred] += 1
    return matrix


def save_confusion_matrix(matrix, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(CIFAR10_CLASSES)),
        yticks=np.arange(len(CIFAR10_CLASSES)),
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)
    device = get_device()

    _, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, dropout=args.dropout).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    labels, preds = collect_predictions(model, test_loader, device)
    matrix = compute_confusion_matrix(labels, preds, num_classes=len(CIFAR10_CLASSES))

    save_path = os.path.join("results", f"{args.exp_name}_confusion_matrix.png")
    save_confusion_matrix(matrix, save_path)
    print(f"Saved confusion matrix: {save_path}")


if __name__ == "__main__":
    main()
