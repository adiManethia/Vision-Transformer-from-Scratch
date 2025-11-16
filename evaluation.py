# evaluation.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   

import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from config import Config as cfg
from data import get_loaders
from model import VisionTransformer

# Create results directory
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def main():
    # -------------------------------------------------
    # 1. Load model
    # -------------------------------------------------
    device = cfg.DEVICE
    model = VisionTransformer().to(device)

    ckpt_path = cfg.BEST_CKPT if cfg.BEST_CKPT.exists() else cfg.LAST_CKPT
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # -------------------------------------------------
    # 2. Full validation metrics
    # -------------------------------------------------
    _, val_loader = get_loaders()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = total_loss / total
    val_acc = 100.0 * correct / total
    print(f"\nValidation → Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}%\n")

    # -------------------------------------------------
    # 3. Plot 10 predictions + SAVE
    # -------------------------------------------------
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    images = images.cpu()
    preds = preds.cpu()
    labels = labels.cpu()

    plt.figure(figsize=(12, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        color = "green" if preds[i] == labels[i] else "red"
        plt.title(f"Pred: {preds[i].item()}\nTrue: {labels[i].item()}", color=color, fontsize=10)
        plt.axis("off")
    plt.suptitle("First 10 Validation Predictions", fontsize=14)
    plt.tight_layout()

    pred_plot_path = RESULTS_DIR / "predictions.png"
    plt.savefig(pred_plot_path, dpi=150, bbox_inches='tight')
    print(f"Predictions saved to: {pred_plot_path}")
    plt.show(block=True)   # ← Keeps window open

    # -------------------------------------------------
    # 4. Plot loss & accuracy curves + SAVE
    # -------------------------------------------------
    if not cfg.HISTORY_FILE.exists():
        print(f"Warning: {cfg.HISTORY_FILE} not found. Run train.py first.")
    else:
        with open(cfg.HISTORY_FILE, "r") as f:
            history = json.load(f)

        epochs = range(1, len(history["train_loss"]) + 1)

        # --- Loss Curve ---
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training & Validation Loss")
        plt.legend(); plt.grid(True); plt.tight_layout()

        loss_plot_path = RESULTS_DIR / "loss_curve.png"
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved to: {loss_plot_path}")
        plt.show(block=True)

        # --- Accuracy Curve ---
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, history["train_acc"], label="Train Acc", marker="o")
        plt.plot(epochs, history["val_acc"], label="Val Acc", marker="s")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Training & Validation Accuracy")
        plt.legend(); plt.grid(True); plt.tight_layout()

        acc_plot_path = RESULTS_DIR / "accuracy_curve.png"
        plt.savefig(acc_plot_path, dpi=150, bbox_inches='tight')
        print(f"Accuracy curve saved to: {acc_plot_path}")
        plt.show(block=True)


if __name__ == '__main__':
    main()