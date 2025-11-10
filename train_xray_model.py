import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from transformers import ViTForImageClassification

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import time
from PIL import Image
import sys

# --- 1. Configuration ---

# Set to True for a quick debug run on a small subset of data
SMOKE_TEST = False 

# --- Paths to your 16-class F-DATASET ---
BASE_DIR = "/home/dheeraj/Documents/Swasth Avishkar Hackathon/F-DATASET"
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
test_dir = os.path.join(BASE_DIR, "test")

# Model & Training Hyperparameters
MODEL_NAME = 'google/vit-base-patch16-224'
NUM_CLASSES = 16   # <-- 16 classes for the X-ray dataset
BATCH_SIZE = 32    # <-- UPDATED for 8GB VRAM
PATIENCE = 5       # Early stopping patience

# Hyperparameters for Feature Extraction (Phase 1)
NUM_EPOCHS_FE = 10
LR_FE = 5e-5

# Hyperparameters for Fine-Tuning (Phase 2)
NUM_EPOCHS_FT = 15
LR_FT = 2e-5

# Saved model paths (new names to avoid overwrites)
MODEL_FE_PATH = "best_model_feature_extraction_f1_8gb.pth"
MODEL_FT_PATH = "best_model_finetuned_f1_8gb.pth"


# --- 2. GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True


# --- 3. Data Loading & Preprocessing ---

# ViT expects 224x224, normalized to mean=0.5, std=0.5
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Robust image loader (skips corrupted files)
def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, IOError) as e:
        print(f"WARNING: Skipping corrupted image {path}: {e}")
        return Image.new('RGB', (224, 224), (0, 0, 0))

# Load datasets with ImageFolder
try:
    train_dataset = ImageFolder(train_dir, transform=train_transform, loader=safe_loader)
    val_dataset = ImageFolder(val_dir, transform=val_transform, loader=safe_loader)
    test_dataset = ImageFolder(test_dir, transform=val_transform, loader=safe_loader)
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure the paths in the script are correct.")
    sys.exit()

# Get class names
class_names = train_dataset.classes
if len(class_names) != NUM_CLASSES:
    print(f"Error: Found {len(class_names)} classes, but NUM_CLASSES is set to {NUM_CLASSES}")
    sys.exit()

print(f"Classes found ({len(class_names)}):")
print(class_names)

# --- 4. Smoke Test Implementation ---
if SMOKE_TEST:
    print("--- RUNNING IN SMOKE TEST MODE ---")
    num_smoke_samples = BATCH_SIZE * 5
    
    train_samples = min(num_smoke_samples, len(train_dataset))
    val_samples = min(num_smoke_samples, len(val_dataset))
    test_samples = min(num_smoke_samples, len(test_dataset))

    train_dataset = Subset(train_dataset, list(range(train_samples)))
    val_dataset = Subset(val_dataset, list(range(val_samples)))
    test_dataset = Subset(test_dataset, list(range(test_samples)))

print(f"Number of train samples: {len(train_dataset)}")
print(f"Number of val samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# --- 5. Loss Function ---
# This dataset was found to be BALANCED, so we use a standard unweighted loss.
print("Using standard (unweighted) CrossEntropyLoss for a balanced dataset.")
criterion = nn.CrossEntropyLoss()


# --- 6. DataLoaders ---
# --- UPDATED: Increased num_workers, set shuffle=True (no sampler) ---
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,  # Shuffle is needed since we're not using a sampler
    num_workers=8, # <-- UPDATED for better CPU/GPU sync
    pin_memory=True
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=8, # <-- UPDATED
    pin_memory=True
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=8, # <-- UPDATED
    pin_memory=True
)


# --- 7. Model Setup ---
def load_model():
    print(f"Loading pre-trained model: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True # Re-initializes the classifier head
    )
    return model.to(device)


# --- 8. Helper Functions: Training, Validation, Plotting ---

def train_one_epoch(model, loader, optimizer, criterion):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion):
    """Validates the model on the validation set."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")

    epoch_loss = val_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def run_training_phase(model, num_epochs, lr, phase_name, model_save_path):
    """Orchestrates a full training phase (FE or FT)."""
    print(f"\n--- Starting {phase_name} Phase ---")
    print(f"Epochs: {num_epochs}, Learning Rate: {lr}")

    if phase_name == "Feature Extraction":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    total_phase_time = time.time() - start_time
    print(f"Finished {phase_name} Phase in {total_phase_time / 60:.2f} minutes.")
    return history

def plot_metrics(history, phase_name):
    """Plots training & validation loss and accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title(f"{phase_name} - Loss")
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title(f"{phase_name} - Accuracy")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"Metrics for {phase_name} (F-DATASET)")
    plt.savefig(f"metrics_{phase_name.lower().replace(' ', '_')}_f1_8gb.png")
    print(f"Saved metrics plot to metrics_{phase_name.lower().replace(' ', '_')}_f1_8gb.png")
    plt.show()

def evaluate_model(model, loader, model_path):
    """Evaluates the model on the test set and prints metrics."""
    print("\n--- Starting Evaluation on Test Set ---")
    
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Evaluation skipped.")
        return
        
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
            images = images.to(device)
            outputs = model(pixel_values=images).logits
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n--- Classification Report ---")
    
    all_label_indices = list(range(len(class_names)))
    
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        labels=all_label_indices,
        zero_division=0
    ))
    
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds, labels=all_label_indices) 
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - Test Set (F-DATASET)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_test_f1_8gb.png")
    print("Saved confusion matrix to confusion_matrix_test_f1_8gb.png")
    plt.show()


# --- 9. Main Execution Pipeline ---
if __name__ == "__main__":
    
    # --- Phase 1: Feature Extraction ---
    model = load_model()
    
    print("Freezing base model layers for feature extraction...")
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
            
    fe_history = run_training_phase(
        model=model,
        num_epochs=NUM_EPOCHS_FE,
        lr=LR_FE,
        phase_name="Feature Extraction",
        model_save_path=MODEL_FE_PATH
    )
    plot_metrics(fe_history, "Feature Extraction")
    
    # --- Phase 2: Fine-Tuning ---
    print("\nLoading best feature extraction model for fine-tuning...")
    try:
        model.load_state_dict(torch.load(MODEL_FE_PATH))
    except FileNotFoundError:
        print(f"Warning: Could not load {MODEL_FE_PATH}. Proceeding with current model weights.")
        
    print("Unfreezing all layers for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True
        
    ft_history = run_training_phase(
        model=model,
        num_epochs=NUM_EPOCHS_FT,
        lr=LR_FT,
        phase_name="Fine-Tuning",
        model_save_path=MODEL_FT_PATH
    )
    plot_metrics(ft_history, "Fine-Tuning")
    
    # --- Phase 3: Evaluation ---
    evaluate_model(model, test_loader, MODEL_FT_PATH)
    
    print("\n--- Pipeline Complete ---")
