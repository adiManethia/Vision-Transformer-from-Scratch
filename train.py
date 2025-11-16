# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from pathlib import Path

from config import Config as cfg
from data import get_loaders
from model import VisionTransformer

torch.manual_seed(cfg.SEED)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbls)
            total_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return total_loss / total, 100.0 * correct / total

def main():
    train_loader, val_loader = get_loaders()
    model = VisionTransformer().to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    best_acc = 0.0

    # History tracking
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.EPOCHS}")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(cfg.DEVICE), lbls.to(cfg.DEVICE)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            preds = outs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
            pbar.set_postfix({"loss": loss.item(), "acc": f"{100.0 * correct / total:.2f}%"})

        # Epoch stats
        train_loss = epoch_loss / total
        train_acc = 100.0 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, cfg.DEVICE)

        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"\n=== Epoch {epoch} ===")
        print(f"Train → loss: {train_loss:.4f} | acc: {train_acc:.2f}%")
        print(f"Val   → loss: {val_loss:.4f} | acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.BEST_CKPT)
            print(f"New best model saved! Acc: {best_acc:.2f}%")

        torch.save(model.state_dict(), cfg.LAST_CKPT)

    # Save training history
    history = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    }
    with open(cfg.HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {cfg.HISTORY_FILE}")

if __name__ == "__main__":
    main()