"""
Task 12: Training Pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
from src.models.multimodal import MultimodalClassifier
from src.data.dataset import get_dataloaders


class Trainer:
    def __init__(self, model, num_classes=4, lr=1e-3, weight_decay=1e-4,
                 patience=10, checkpoint_dir="checkpoints"):
        self.model = model
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.criterion = nn.CrossEntropyLoss()

        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for img, clin, labels in loader:
            img = img.to(self.device)
            clin = clin.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(img, clin)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * img.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        for img, clin, labels in loader:
            img = img.to(self.device)
            clin = clin.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(img, clin)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * img.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    def save_checkpoint(self, epoch, val_loss, filename="best_model.pth"):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)

    def fit(self, train_loader, val_loader, num_epochs=50):
        print(f"Training on {self.device} for up to {num_epochs} epochs")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        for epoch in range(num_epochs):
            start = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.scheduler.step(val_loss)

            elapsed = time.time() - start
            lr = self.optimizer.param_groups[0]['lr']

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} ({elapsed:.1f}s) - "
                  f"lr={lr:.2e} - "
                  f"train_loss={train_loss:.4f} acc={train_acc:.3f} - "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.counter = 0
                self.save_checkpoint(epoch + 1, val_loss)
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining complete. Best epoch: {self.best_epoch}, Best val_loss: {self.best_val_loss:.4f}")
        return self.history


if __name__ == "__main__":
    print("Creating model...")
    model = MultimodalClassifier(num_classes=4, dropout=0.3)

    print("Loading data...")
    loaders, feat_names, num_classes = get_dataloaders(batch_size=2)

    trainer = Trainer(model, num_classes=num_classes, lr=1e-3, patience=10)
    history = trainer.fit(loaders["train"], loaders["val"], num_epochs=3)

    print(f"\nFinal train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("Training pipeline test passed!")
