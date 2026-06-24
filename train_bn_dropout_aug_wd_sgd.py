import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN_BN_Dropout

from train_bn_dropout_aug_wd import (
    set_seed,
    get_data_loaders_aug,
    train_one_epoch,
    evaluate,
    save_csv,
    save_curves,
    save_summary,
)


def main():
    set_seed(42)

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    exp_name = "bn_dropout_aug_wd_sgd_momentum"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    batch_size = 128
    num_workers = 4
    epochs = 50

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    dropout = 0.3

    train_loader, test_loader = get_data_loaders_aug(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = CNN_BN_Dropout(num_classes=10, dropout=dropout).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
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
                    "momentum": momentum,
                    "augmentation": True,
                    "optimizer": "SGD_Momentum",
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
        optimizer_name="SGD + Momentum",
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
