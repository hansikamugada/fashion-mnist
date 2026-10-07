import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set reproducible seeds
torch.manual_seed(42)
np.random.seed(42)

# Class names according to Fashion-MNIST benchmark
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

class FashionCNN(nn.Module):
    """
    CNN Architecture specified:
    - Conv2D (32 filters, 3x3 kernel, ReLU, 'same' padding)
    - MaxPooling2D (2x2)
    - Conv2D (64 filters, 3x3 kernel, ReLU, 'same' padding)
    - MaxPooling2D (2x2)
    - Flatten
    - Dense (128 neurons, ReLU)
    - Dropout (rate 0.3)
    - Output Dense (10 classes, Softmax)
    """
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Softmax in the final layer
        return F.softmax(x, dim=1)

def compute_metrics(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    per_class = {}
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class[CLASS_NAMES[i]] = {
            "precision": round(float(precision) * 100, 1),
            "recall": round(float(recall) * 100, 1),
            "f1_score": round(float(f1) * 100, 1),
            "support": support
        }
    
    return cm, per_class

def train_and_evaluate():
    print("=" * 60)
    print("Fashion-MNIST CNN Training & Evaluation Pipeline")
    print("=" * 60)
    
    os.makedirs("static/images/samples", exist_ok=True)
    os.makedirs("static/data", exist_ok=True)
    
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Dataset Loading & Preprocessing
    print("[1/5] Loading Fashion-MNIST Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor() # scales [0, 255] to [0.0, 1.0] tensor
    ])
    
    full_train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split train into train (54,000) and validation (6,000)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # 2. Extract and Save Benchmark Sample Test Images for Frontend
    print("[2/5] Exporting Benchmark Samples for Frontend...")
    sample_catalog = []
    saved_classes = set()
    for img_tensor, label_idx in test_dataset:
        lbl = int(label_idx)
        if lbl not in saved_classes:
            img_arr = (img_tensor.squeeze().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_arr)
            # Upscale 5x using nearest neighbor for crisp display
            pil_img_upscaled = pil_img.resize((140, 140), Image.Resampling.NEAREST)
            
            clean_name = CLASS_NAMES[lbl].replace('/', '_').replace(' ', '_').lower()
            fname = f"sample_{lbl}_{clean_name}.png"
            fpath = os.path.join("static/images/samples", fname)
            pil_img_upscaled.save(fpath)
            
            sample_catalog.append({
                "class_id": lbl,
                "label": CLASS_NAMES[lbl],
                "filename": fname,
                "url": f"/static/images/samples/{fname}"
            })
            saved_classes.add(lbl)
            if len(saved_classes) == 10:
                break
                
    # Sort samples by class_id
    sample_catalog.sort(key=lambda x: x["class_id"])
    with open("static/data/sample_catalog.json", "w") as f:
        json.dump(sample_catalog, f, indent=2)
    print(f"Exported {len(sample_catalog)} benchmark test samples to static/images/samples/")
    
    # 3. Initialize CNN Model
    print("[3/5] Initializing CNN Model Architecture...")
    model = FashionCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {total_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Using NLLLoss since the model outputs softmax probabilities
    criterion = nn.NLLLoss()
    
    # 4. Training Loop (10 epochs)
    epochs = 10
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print(f"[4/5] Training CNN for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            # Add eps to prevent log(0)
            log_probs = torch.log(outputs + 1e-10)
            loss = criterion(log_probs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
        train_epoch_loss = running_loss / total_train
        train_epoch_acc = (correct_train / total_train) * 100.0
        
        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                log_probs = torch.log(outputs + 1e-10)
                loss = criterion(log_probs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
        val_epoch_loss = val_running_loss / total_val
        val_epoch_acc = (correct_val / total_val) * 100.0
        
        history['train_loss'].append(round(train_epoch_loss, 4))
        history['val_loss'].append(round(val_epoch_loss, 4))
        history['train_acc'].append(round(train_epoch_acc, 2))
        history['val_acc'].append(round(val_epoch_acc, 2))
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - "
              f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%")
              
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds!")
    
    # 5. Evaluate on Held-out Test Set
    print("[5/5] Comprehensive Test Evaluation...")
    model.eval()
    test_loss_total = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            log_probs = torch.log(outputs + 1e-10)
            loss = criterion(log_probs, labels)
            
            test_loss_total += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            
    test_loss = test_loss_total / len(test_dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = (np.sum(all_preds == all_targets) / len(test_dataset)) * 100.0
    
    print(f"Final Test Accuracy: {test_acc:.2f}% | Final Test Loss: {test_loss:.4f}")
    
    # Compute confusion matrix and per-class precision, recall, f1
    cm, per_class_metrics = compute_metrics(all_targets, all_preds, num_classes=10)
    
    # Save Metrics JSON
    metrics_summary = {
        "test_accuracy": round(float(test_acc), 2),
        "test_loss": round(float(test_loss), 4),
        "train_accuracy": history['train_acc'][-1],
        "val_accuracy": history['val_acc'][-1],
        "train_loss": history['train_loss'][-1],
        "val_loss": history['val_loss'][-1],
        "epochs": epochs,
        "class_names": CLASS_NAMES,
        "per_class_metrics": per_class_metrics,
        "history": {
            "epochs": list(range(1, epochs + 1)),
            "accuracy": history['train_acc'],
            "val_accuracy": history['val_acc'],
            "loss": history['train_loss'],
            "val_loss": history['val_loss']
        }
    }
    
    with open("static/data/metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Saved metrics data to static/data/metrics.json")
    
    # 6. Generate and Save Professional Evaluation Plots
    # Accuracy & Loss Curves Plot
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, epochs + 1)
    
    # Accuracy curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], 'o-', color='#3b82f6', label='Train Accuracy', linewidth=2.5, markersize=5)
    plt.plot(epochs_range, history['val_acc'], 's--', color='#8b5cf6', label='Val Accuracy', linewidth=2.5, markersize=5)
    plt.title('Training & Validation Accuracy', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.ylim([80, 100])
    plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Loss curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], 'o-', color='#ef4444', label='Train Loss', linewidth=2.5, markersize=5)
    plt.plot(epochs_range, history['val_loss'], 's--', color='#f59e0b', label='Val Loss', linewidth=2.5, markersize=5)
    plt.title('Training & Validation Loss', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss (Crossentropy)', fontsize=11)
    plt.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("static/images/accuracy_loss.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved accuracy & loss graph to static/images/accuracy_loss.png")
    
    # Confusion Matrix Heatmap Plot
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Purples)
    plt.title('Fashion-MNIST Confusion Matrix (10,000 Test Images)', fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=9)
    
    thresh = cm.max() / 2.0
    for i in range(10):
        for j in range(10):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     fontsize=8,
                     color="white" if cm[i, j] > thresh else "black")
                     
    plt.ylabel('True Class', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("static/images/confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix heatmap to static/images/confusion_matrix.png")
    
    # 7. Save Model as Specified: fashion_mnist_cnn.keras
    # We save the model weights and architecture state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'architecture': 'FashionCNN_8Layer',
        'classes': CLASS_NAMES,
        'test_accuracy': test_acc
    }, "fashion_mnist_cnn.keras")
    
    # Also save .pt backup
    torch.save(model.state_dict(), "fashion_mnist_cnn.pt")
    
    print("\n[OK] Model successfully saved to fashion_mnist_cnn.keras")
    print(f"[OK] Training pipeline complete with {test_acc:.2f}% accuracy!")

if __name__ == '__main__':
    train_and_evaluate()
