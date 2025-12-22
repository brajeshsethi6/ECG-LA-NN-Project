
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        
        # Balanced Class Weights (Approximate inverse frequency)
        # N: 1.0 (Majority)
        # S: ~30.0 
        # V: ~15.0
        # F: ~100.0 (Minority)
        # Q: ~15.0
        class_weights = torch.tensor([1.0, 30.0, 15.0, 100.0, 15.0]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.LEARNING_RATE, 
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
        
        self.model.to(self.device)

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _ = self.model(signals)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Gradient Clipping to prevent explosion (common in LNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(self.train_loader), 'acc': 100 * correct / total})
            
        return running_loss / len(self.train_loader), 100 * correct / total

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for signals, labels in self.val_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs, _ = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        loss = running_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} [Val]: Loss: {loss:.4f}, Acc: {acc:.2f}%")
        return loss, acc

    def train(self):
        print(f"Starting training on {self.device}...")
        best_acc = 0.0
        
        for epoch in range(self.config.NUM_EPOCHS):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            
            self.scheduler.step()
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"{self.config.MODELS_DIR}/la_nn_best.pth")
                print("Saved new best model.")
                
        print("Training complete.")
