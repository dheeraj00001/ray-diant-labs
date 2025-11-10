import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from transformers import AutoModelForImageClassification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import time
from PIL import Image
import sys
import json
import glob
from datetime import datetime
import shutil

# Configuration
SMOKE_TEST = False

# Local dataset path - UPDATE THIS PATH
BASE_DIR = "/home/neeraj/Documents/Projects/CT SCAN Dataset/F-3-DATASET"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

# Model & Training Hyperparameters
MODEL_NAME = 'microsoft/swin-base-patch4-window7-224'
NUM_CLASSES = 9
BATCH_SIZE = 16  # REDUCED from 32 to prevent OOM
PATIENCE = 7  # Increased for more robust training

# Hyperparameters for Feature Extraction (Phase 1)
NUM_EPOCHS_FE = 20
LR_FE = 2e-5  # Slightly lower for better stability

# Hyperparameters for Fine-Tuning (Phase 2)
NUM_EPOCHS_FT = 5
LR_FT = 1e-5

# Checkpoint directory
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model save paths (local)
MODEL_FE_PATH = "./best_model_feature_extraction_f3.pth"
MODEL_FT_PATH = "./best_model_swin_f3.pth"

print(f"Base data directory: {BASE_DIR}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Final model will be saved to: {MODEL_FT_PATH}")
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

# GPU Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Enable mixed precision training for RTX 4050 (using updated API)
    scaler = torch.amp.GradScaler('cuda')
    print("Mixed precision training enabled")
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Data Loading & Preprocessing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f"WARNING: Skipping corrupted image {path}: {e}")
        return Image.new('RGB', (224, 224), (0, 0, 0))

# Load datasets
try:
    train_dataset = ImageFolder(train_dir, transform=train_transform, loader=safe_loader)
    val_dataset = ImageFolder(val_dir, transform=val_transform, loader=safe_loader)
    test_dataset = ImageFolder(test_dir, transform=val_transform, loader=safe_loader)
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure that dataset paths are correct.")
    sys.exit()

# Get class names
class_names = train_dataset.classes
if len(class_names) != NUM_CLASSES:
    print(f"Error: Found {len(class_names)} classes, but NUM_CLASSES is set to {NUM_CLASSES}")
    sys.exit()

print(f"\nClasses found ({len(class_names)}):")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

# Smoke Test Implementation
if SMOKE_TEST:
    print("--- RUNNING IN SMOKE TEST MODE ---")
    num_smoke_samples = BATCH_SIZE * 5
    train_samples = min(num_smoke_samples, len(train_dataset))
    val_samples = min(num_smoke_samples, len(val_dataset))
    test_samples = min(num_smoke_samples, len(test_dataset))

    train_dataset = Subset(train_dataset, list(range(train_samples)))
    val_dataset = Subset(val_dataset, list(range(val_samples)))
    test_dataset = Subset(test_dataset, list(range(test_samples)))

print(f"\nDataset sizes:")
print(f"Train: {len(train_dataset)} samples")
print(f"Validation: {len(val_dataset)} samples")
print(f"Test: {len(test_dataset)} samples")

# Class Imbalance Handling
print("\n--- Handling Class Imbalance with WeightedRandomSampler ---")

if not SMOKE_TEST:
    targets = np.array(train_dataset.targets)
    class_counts = np.array([(targets == i).sum() for i in range(NUM_CLASSES)])

    if (class_counts == 0).any():
        print("Warning: Some classes have 0 samples in the training set!")

    print("Class counts (train):", class_counts)

    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights[class_weights == float('inf')] = 0

    sample_weights = torch.tensor([class_weights[t] for t in targets])

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("WeightedRandomSampler enabled.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
else:
    sampler = None
    print("Smoke test: Using uniform sampling and unweighted loss.")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# DataLoaders - Optimized for RTX 4050 with memory constraints
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,  # REDUCED BATCH SIZE
    sampler=sampler,
    shuffle=(sampler is None),
    num_workers=4,  # REDUCED FROM 8
    pin_memory=True,
    drop_last=True,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,  # REDUCED BATCH SIZE
    shuffle=False,
    num_workers=4,  # REDUCED FROM 8
    pin_memory=True,
    persistent_workers=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,  # REDUCED BATCH SIZE
    shuffle=False,
    num_workers=4,  # REDUCED FROM 8
    pin_memory=True,
    persistent_workers=True
)

# Model Setup
def load_model():
    print(f"\nLoading pre-trained model: {MODEL_NAME}")
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    
    # Add dropout to classifier
    if hasattr(model.classifier, 'dropout'):
        model.classifier.dropout.p = 0.5
    else:
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            model.classifier
        )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model.to(device)

# Checkpoint functions
def save_checkpoint(model, optimizer, scheduler, epoch, history, phase_name, best_val_loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history,
        'best_val_loss': best_val_loss,
        'phase_name': phase_name
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_latest.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # Save epoch checkpoint
    epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_epoch_{epoch}.pth")
    torch.save(checkpoint, epoch_checkpoint_path)
    
    # Save best model
    if is_best:
        best_model_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_best.pth")
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved with validation loss: {best_val_loss:.4f}")
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['history'], checkpoint['best_val_loss']

def find_latest_checkpoint(phase_name):
    pattern = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_latest.pth")
    if os.path.exists(pattern):
        return pattern
    return None

# Memory management function
def clear_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Training Functions with Advanced Progress Bars and Memory Management
def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, epoch=0, total_epochs=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar with detailed information
    with tqdm(total=len(loader), 
              desc=f"Epoch {epoch+1}/{total_epochs}", 
              unit="batch",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            
            # Mixed precision training (using updated API)
            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=images).logits
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Only step scheduler if it's not OneCycleLR (handled at epoch level)
            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                scheduler.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar with detailed metrics
            current_acc = correct / total
            current_loss = running_loss / total
            current_lr = optimizer.param_groups[0]["lr"]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{current_loss:.4f}',
                'acc': f'{100.*current_acc:.2f}%',
                'lr': f'{current_lr:.2e}',
                'mem': f'{torch.cuda.memory_allocated()/1e9:.2f}GB'
            })
            pbar.update(1)
            
            # Clear memory more frequently
            if batch_idx % 50 == 0:
                clear_memory()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with tqdm(total=len(loader), 
              desc="Validating", 
              unit="batch",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=images).logits
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                current_acc = correct / total
                current_loss = val_loss / total
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{current_loss:.4f}',
                    'acc': f'{100.*current_acc:.2f}%'
                })
                pbar.update(1)
    
    epoch_loss = val_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def run_training_phase(model, num_epochs, lr, phase_name, model_save_path, resume=False):
    print(f"\n--- Starting {phase_name} Phase ---")
    print(f"Epochs: {num_epochs}, Learning Rate: {lr}")
    
    # Initialize history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Check for checkpoint to resume from
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume:
        checkpoint_path = find_latest_checkpoint(phase_name)
        if checkpoint_path:
            if phase_name == "Feature Extraction":
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                      lr=lr, weight_decay=0.05)
            else:
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
            
            # Create a new OneCycleLR scheduler with adjusted total steps
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=lr,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy='cos'
            )
            
            start_epoch, history, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            print(f"Resuming from epoch {start_epoch+1}")
            
            # Adjusted OneCycleLR scheduler to account for already completed steps
            if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                completed_steps = start_epoch * len(train_loader)
                total_steps = num_epochs * len(train_loader)
                remaining_steps = total_steps - completed_steps
                
                # Create a new scheduler with remaining steps
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=lr,
                    total_steps=remaining_steps,
                    pct_start=0.1,
                    anneal_strategy='cos'
                )
                print(f"Adjusted OneCycleLR scheduler with {remaining_steps} remaining steps")
        else:
            print("No checkpoint found, starting from scratch")
            resume = False
    
    if not resume:
        if phase_name == "Feature Extraction":
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=lr, weight_decay=0.05)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        # Memory cleanup
        clear_memory()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, epoch, num_epochs)
        
        # Step's OneCycleLR scheduler after each epoch
        if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            for _ in range(len(train_loader)):
                scheduler.step()
        
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience {patience_counter}/{PATIENCE}")
        
        # Save checkpoint for every epoch
        save_checkpoint(model, optimizer, scheduler, epoch, history, phase_name, best_val_loss, is_best)
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
    
    total_phase_time = time.time() - start_time
    print(f"\nFinished {phase_name} Phase in {total_phase_time / 60:.2f} minutes.")
    
    # Load best model for this phase
    best_model_path = os.path.join(CHECKPOINT_DIR, f"{phase_name.lower().replace(' ', '_')}_best.pth")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Save final model to specified path
    torch.save(model.state_dict(), model_save_path)
    
    return history

def plot_metrics(history, phase_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title(f"{phase_name} - Loss", fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_title(f"{phase_name} - Accuracy", fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Metrics for {phase_name} (F-3-DATASET)", fontsize=16)
    plt.tight_layout()
    
    plot_filename = f"metrics_{phase_name.lower().replace(' ', '_')}_f3.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved metrics plot to {plot_filename}")
    plt.show()

def evaluate_model(model, loader, model_path):
    print("\n--- Starting Evaluation on Test Set ---")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Evaluation skipped.")
        return
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with tqdm(total=len(loader), 
              desc="Testing", 
              unit="batch",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(pixel_values=images).logits
                    preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.update(1)
    
    print("\n--- Classification Report ---")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    ))
    
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Test Set (F-3-DATASET)", fontsize=16)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    
    cm_filename = "confusion_matrix_test_f3.png"
    plt.savefig(cm_filename, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {cm_filename}")
    plt.show()

# Main Execution Pipeline
if __name__ == "__main__":
    # Phase 1: Feature Extraction
    model = load_model()
    
    print("\nFreezing base model layers for feature extraction...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # Check if we should resume training
    resume_fe = os.path.exists(os.path.join(CHECKPOINT_DIR, "feature_extraction_latest.pth"))
    
    fe_history = run_training_phase(
        model=model,
        num_epochs=NUM_EPOCHS_FE,
        lr=LR_FE,
        phase_name="Feature Extraction",
        model_save_path=MODEL_FE_PATH,
        resume=resume_fe
    )
    plot_metrics(fe_history, "Feature Extraction")
    
    # Phase 2: Fine-Tuning
    print(f"\nLoading best feature extraction model ({MODEL_FE_PATH}) for fine-tuning...")
    try:
        model.load_state_dict(torch.load(MODEL_FE_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Warning: Could not load {MODEL_FE_PATH}. Proceeding with current model weights.")
    
    print("\nUnfreezing all layers for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
    
    # Check if we should resume fine-tuning
    resume_ft = os.path.exists(os.path.join(CHECKPOINT_DIR, "fine-tuning_latest.pth"))
    
    ft_history = run_training_phase(
        model=model,
        num_epochs=NUM_EPOCHS_FT,
        lr=LR_FT,
        phase_name="Fine-Tuning",
        model_save_path=MODEL_FT_PATH,
        resume=resume_ft
    )
    plot_metrics(ft_history, "Fine-Tuning")
    
    # Phase 3: Evaluation
    evaluate_model(model, test_loader, MODEL_FT_PATH)
    
    print("\n--- Pipeline Complete ---")
    print(f"The best model has been saved to: {MODEL_FT_PATH}")
