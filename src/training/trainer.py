
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
from datetime import datetime
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, config, train_loader, val_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.DEVICE
        
        # Setup logging directories
        self.log_dir = os.path.join(config.BASE_DIR, '..', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')
        self.metrics_file = os.path.join(self.log_dir, f'metrics_{timestamp}.csv')
        
        # Initialize metrics CSV
        with open(self.metrics_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                           'val_acc_N', 'val_acc_S', 'val_acc_V', 'val_acc_F', 'val_acc_Q',
                           'learning_rate', 'time_elapsed'])
        
        self.log(f"Training started at {timestamp}")
        self.log(f"Device: {self.device}")
        self.log(f"Log file: {self.log_file}")
        self.log(f"Metrics file: {self.metrics_file}")
        
        # Balanced Class Weights (Approximate inverse frequency)
        # N: 1.0 (Majority)
        # S: ~30.0 
        # V: ~15.0
        # F: ~100.0 (Minority)
        # Q: ~15.0
        class_weights = torch.tensor([1.0, 30.0, 15.0, 100.0, 15.0]).to(self.device)
        self.log(f"Class weights: {class_weights.cpu().numpy()}")
        
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
        self.start_time = time.time()
    
    def log(self, message):
        """Write message to both console and log file"""
        # Encode for console print if needed, but usually modern terminals handle it
        print(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} [Train]")
        for batch_idx, (signals, labels) in enumerate(pbar):
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
            
            pbar.set_postfix({'loss': running_loss/(batch_idx+1), 'acc': 100 * correct / total})
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        self.log(f"Epoch {epoch+1} [Train]: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        return epoch_loss, epoch_acc

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class tracking
        class_correct = [0] * self.config.NUM_CLASSES
        class_total = [0] * self.config.NUM_CLASSES
        
        with torch.no_grad():
            for signals, labels in self.val_loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs, _ = self.model(signals)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
        
        acc = 100 * correct / total
        loss = running_loss / len(self.val_loader)
        
        # Calculate per-class accuracies
        class_accs = []
        for i in range(self.config.NUM_CLASSES):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accs.append(class_acc)
                self.log(f"  Class {self.config.AAMI_CLASSES[i]}: {class_acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                class_accs.append(0.0)
                self.log(f"  Class {self.config.AAMI_CLASSES[i]}: N/A (no samples)")
        
        self.log(f"Epoch {epoch+1} [Val]: Loss: {loss:.4f}, Overall Acc: {acc:.2f}%")
        
        return loss, acc, class_accs

    def train(self):
        self.log(f"Starting training on {self.device}...")
        self.log(f"Total epochs: {self.config.NUM_EPOCHS}")
        self.log(f"Batch size: {self.config.BATCH_SIZE}")
        self.log(f"Learning rate: {self.config.LEARNING_RATE}")
        self.log(f"Training samples: {len(self.train_loader.dataset)}")
        self.log(f"Validation samples: {len(self.val_loader.dataset)}")
        self.log("="*60)
        
        best_acc = 0.0
        best_epoch = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, class_accs = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Time tracking
            epoch_time = time.time() - epoch_start
            total_time = time.time() - self.start_time
            
            # Save metrics to CSV
            with open(self.metrics_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, 
                    f"{train_loss:.4f}", 
                    f"{train_acc:.2f}", 
                    f"{val_loss:.4f}", 
                    f"{val_acc:.2f}",
                    f"{class_accs[0]:.2f}",  # N
                    f"{class_accs[1]:.2f}",  # S
                    f"{class_accs[2]:.2f}",  # V
                    f"{class_accs[3]:.2f}",  # F
                    f"{class_accs[4]:.2f}",  # Q
                    f"{current_lr:.6f}",
                    f"{total_time:.1f}"
                ])
            
            self.log(f"Learning Rate: {current_lr:.6f}")
            self.log(f"Epoch Time: {epoch_time:.1f}s, Total Time: {total_time:.1f}s")
            self.log("="*60)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch + 1
                model_path = f"{self.config.MODELS_DIR}/la_nn_best.pth"
                torch.save(self.model.state_dict(), model_path)
                self.log(f"✓ Saved new best model (Acc: {best_acc:.2f}%) to {model_path}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"{self.config.MODELS_DIR}/la_nn_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, checkpoint_path)
                self.log(f"✓ Saved checkpoint to {checkpoint_path}")
        
        self.log("="*60)
        self.log(f"Training complete!")
        self.log(f"Best validation accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
        self.log(f"Total training time: {(time.time() - self.start_time):.1f}s")
        self.log(f"Logs saved to: {self.log_file}")
        self.log(f"Metrics saved to: {self.metrics_file}")
