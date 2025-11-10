import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from PIL import Image, ImageFile
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import gc

warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Configuration ---
ROOT_DIR = '/home/dheeraj/Documents/Swasth Avishkar Hackathon/Base_Model_Folder/Dataset_preprocessed'
TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
VAL_DIR = os.path.join(ROOT_DIR, 'val')
TEST_DIR = os.path.join(ROOT_DIR, 'test')
SAVE_DIR = os.path.join(ROOT_DIR, 'model_checkpoints_robust') # Saving to a new directory
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
IMG_SIZE = 224
DROPOUT_RATE = 0.5
NUM_CLASSES = 3
NUM_WORKERS = 2
NUM_EPOCHS_FEATURE = 2
NUM_EPOCHS_FINE = 3

# --- Smoke Test (for quick debugging) ---
SMOKE_TEST = False
SMOKE_SUBSET_SIZE = 100

# --- Data Transforms ---
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- Dataset Analysis ---
print("=== Dataset Analysis ===")
analyze_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=analyze_transform)
val_dataset_full = datasets.ImageFolder(VAL_DIR, transform=analyze_transform)
test_dataset_full = datasets.ImageFolder(TEST_DIR, transform=analyze_transform)

print(f"Dataset Splits (full):")
print(f"Train: {len(train_dataset_full)} samples")
print(f"Val: {len(val_dataset_full)} samples")
print(f"Test: {len(test_dataset_full)} samples")

class_names = train_dataset_full.classes
print(f"Classes Found: {class_names}")

def analyze_classes(dataset, split_name):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(dataset)
    print(f"\n{split_name} Class Distribution:")
    for cls, count in enumerate(class_counts):
        perc = (count / total) * 100
        print(f"Class {cls} ({class_names[cls]}): {count} samples ({perc:.2f}%)")
    return class_counts

train_class_counts = analyze_classes(train_dataset_full, "Train")
val_class_counts = analyze_classes(val_dataset_full, "Val")
test_class_counts = analyze_classes(test_dataset_full, "Test")


# --- Robust Imbalance Handling ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            total_samples = float(sum(train_class_counts))
            alpha_weights = total_samples / (NUM_CLASSES * torch.tensor(train_class_counts, dtype=torch.float))
            self.alpha = alpha_weights.to(device)
        else:
            self.alpha = alpha.to(device)
            
        self.gamma = gamma
        self.reduction = reduction
        print(f"FocalLoss initialized with alpha={self.alpha.cpu().numpy()} and gamma={self.gamma}")

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha[targets] * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# --- Data Augmentation and Loaders ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    normalize
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    normalize
])

train_dataset_full.transform = train_transform
val_dataset_full.transform = val_test_transform
test_dataset_full.transform = val_test_transform


if SMOKE_TEST:
    sampler = None
    train_dataset = Subset(train_dataset_full, np.random.choice(len(train_dataset_full), SMOKE_SUBSET_SIZE, replace=False))
else:
    train_dataset = train_dataset_full
    labels = np.array(train_dataset.targets)
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight_per_sample = 1.0 / class_sample_count
    samples_weight = np.array([weight_per_sample[t] for t in labels])
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).double(), len(samples_weight))


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset_full, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# --- Model Definition ---
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(DROPOUT_RATE),
    nn.Linear(num_ftrs, NUM_CLASSES)
)
model = model.to(device)

# --- Loss, Optimizer, Scheduler ---
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# CORRECTED LINE: Removed the 'verbose' argument
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)

# --- Callbacks for Early Stopping and Checkpointing ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_metric, model):
        if self.best_metric is None:
            self.best_metric = val_metric
            self.save_checkpoint(model)
        elif val_metric > self.best_metric + self.min_delta:
            self.best_metric = val_metric
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered.")
                if self.restore_best_weights:
                    print(f"Restoring model weights from the best F1-score: {self.best_metric:.4f}")
                    model.load_state_dict(self.best_weights)
                return True
        return False

    def save_checkpoint(self, model):
        # Save weights to CPU to avoid potential GPU memory issues
        self.best_weights = OrderedDict((k, v.cpu()) for k, v in model.state_dict().items())


class ModelCheckpoint:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_metric = -1

    def save(self, model, epoch, val_metric, optimizer):
        if val_metric > self.best_metric:
            self.best_metric = val_metric
            os.makedirs(self.save_path, exist_ok=True)
            checkpoint_path = os.path.join(self.save_path, f'best_model_epoch_{epoch}_f1_{val_metric:.4f}.pth')
            print(f"Validation F1 improved. Saving checkpoint to {checkpoint_path}...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metric,
            }, checkpoint_path)

# --- Training and Evaluation Loop ---
def train_val_epoch(model, loader, optimizer, criterion, phase='train'):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"{phase.capitalize()} Epoch")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if is_train:
                loss.backward()
                optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    epoch_loss = running_loss / len(loader.dataset)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\n{phase.capitalize()} Per-Class Report:\n{classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)}")
    
    return epoch_loss, f1_macro


# --- Main Execution ---
def main():
    # === PHASE 1: FEATURE EXTRACTION ===
    print("\n=== Feature Extraction Phase (Frozen Backbone) ===")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer_fe = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler_fe = ReduceLROnPlateau(optimizer_fe, mode='max', factor=0.2, patience=3)
    
    early_stopping_fe = EarlyStopping(patience=5, min_delta=0.001)
    checkpoint_fe = ModelCheckpoint(SAVE_DIR)
    
    for epoch in range(NUM_EPOCHS_FEATURE):
        print(f"\n--- Starting FE Epoch {epoch+1}/{NUM_EPOCHS_FEATURE} ---")
        train_loss, train_f1 = train_val_epoch(model, train_loader, optimizer_fe, criterion, 'train')
        val_loss, val_f1 = train_val_epoch(model, val_loader, optimizer_fe, criterion, 'val')
        
        print(f"FE Epoch {epoch+1}: Train Loss={train_loss:.4f}, F1={train_f1:.4f} | Val Loss={val_loss:.4f}, F1={val_f1:.4f}")
        
        scheduler_fe.step(val_f1)
        checkpoint_fe.save(model, epoch, val_f1, optimizer_fe)
        if early_stopping_fe(val_f1, model):
            break
            
    # === PHASE 2: FINE-TUNING ===
    print("\n=== Fine-Tuning Phase (Unfrozen Layers) ===")
    if early_stopping_fe.best_weights:
        model.load_state_dict(early_stopping_fe.best_weights)
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(model.parameters(), lr=LEARNING_RATE / 10, weight_decay=1e-4)
    scheduler_ft = ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.2, patience=5)
    
    early_stopping_ft = EarlyStopping(patience=10, min_delta=0.001)
    checkpoint_ft = ModelCheckpoint(SAVE_DIR)
    
    for epoch in range(NUM_EPOCHS_FINE):
        print(f"\n--- Starting FT Epoch {epoch+1}/{NUM_EPOCHS_FINE} ---")
        train_loss, train_f1 = train_val_epoch(model, train_loader, optimizer_ft, criterion, 'train')
        val_loss, val_f1 = train_val_epoch(model, val_loader, optimizer_ft, criterion, 'val')
        
        print(f"FT Epoch {epoch+1}: Train Loss={train_loss:.4f}, F1={train_f1:.4f} | Val Loss={val_loss:.4f}, F1={val_f1:.4f}")
        
        scheduler_ft.step(val_f1)
        checkpoint_ft.save(model, epoch + NUM_EPOCHS_FEATURE, val_f1, optimizer_ft)
        if early_stopping_ft(val_f1, model):
            break

    # --- FINAL EVALUATION ---
    print("\n=== Final Test Evaluation ===")
    if early_stopping_ft.best_weights:
        model.load_state_dict(early_stopping_ft.best_weights)
    else: # Fallback to FE weights if FT didn't run or improve
        model.load_state_dict(early_stopping_fe.best_weights)

    evaluate_model(model, test_loader, 'test', class_names)
    
    final_path = os.path.join(SAVE_DIR, 'final_best_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final best model saved to {final_path}")


def evaluate_model(model, loader, phase, class_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"{phase.capitalize()} Eval"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(f"\n--- {phase.capitalize()} Final Metrics ---")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    with open(os.path.join(SAVE_DIR, 'final_test_metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{phase.capitalize()} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(SAVE_DIR, f'{phase}_confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    main()