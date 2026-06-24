import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ensure_project_dirs, get_data_loaders
from model import build_model
from utils import (
    evaluate,
    get_device,
    save_csv,
    save_curves,
    save_summary,
    set_seed,
    train_one_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 experiments.")
    parser.add_argument("--model", choices=["baseline", "cnn_bn_dropout"], required=True)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "sgd_momentum"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd_momentum":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def main():
    args = parse_args()
    ensure_project_dirs()
    set_seed(args.seed)
    device = get_device()

    train_loader, test_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
    )
    model = build_model(args.model, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)

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
    checkpoint_path = os.path.join("checkpoints", f"{args.exp_name}_best.pth")
    csv_path = os.path.join("results", f"{args.exp_name}_log.csv")
    curves_path = os.path.join("results", f"{args.exp_name}_curves.png")
    summary_path = os.path.join("results", f"{args.exp_name}_summary.txt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
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
                    "model": args.model,
                    "augment": args.augment,
                    "optimizer": args.optimizer,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                },
                checkpoint_path,
            )

    total_time = time.time() - start_time
    save_csv(history, csv_path)
    save_curves(history, curves_path)
    save_summary(summary_path, args, best_acc, best_epoch, total_time)

    print("=" * 60)
    print(f"Best Test Acc: {best_acc * 100:.2f}% at epoch {best_epoch}")
    print(f"Total Time: {total_time / 60:.2f} min")
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved log: {csv_path}")
    print(f"Saved curves: {curves_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
