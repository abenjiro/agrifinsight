"""
Train ML model for crop recommendation based on geospatial data
Uses PyTorch neural network with custom architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class CropDataset(Dataset):
    """PyTorch Dataset for crop recommendation"""

    def __init__(self, df: pd.DataFrame, scaler=None, fit_scaler=False):
        # Features
        self.features = df[[
            'latitude', 'longitude', 'altitude',
            'avg_temperature', 'avg_annual_rainfall', 'soil_ph',
            'soil_type_encoded', 'climate_zone_encoded', 'terrain_type_encoded'
        ]].values.astype(np.float32)

        # Target (crop label)
        self.labels = df['crop_label'].values.astype(np.int64)

        # Suitability scores (for additional supervision)
        self.suitability = df['suitability_score'].values.astype(np.float32)

        # Normalize features
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.LongTensor([self.labels[idx]])[0],
            torch.FloatTensor([self.suitability[idx]])[0]
        )


class CropRecommendationModel(nn.Module):
    """
    Neural network for crop recommendation
    Architecture: Input -> Hidden Layers -> Crop Classification + Suitability Regression
    """

    def __init__(self, input_size=9, num_crops=6, hidden_sizes=[128, 64, 32]):
        super(CropRecommendationModel, self).__init__()

        # Shared layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_size = hidden_size

        self.shared_layers = nn.Sequential(*layers)

        # Classification head (crop prediction)
        self.classifier = nn.Sequential(
            nn.Linear(prev_size, num_crops)
        )

        # Regression head (suitability score prediction)
        self.regressor = nn.Sequential(
            nn.Linear(prev_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output between 0 and 1, will be scaled to 0-100
        )

    def forward(self, x):
        # Shared feature extraction
        features = self.shared_layers(x)

        # Crop classification
        crop_logits = self.classifier(features)

        # Suitability regression
        suitability = self.regressor(features) * 100  # Scale to 0-100

        return crop_logits, suitability.squeeze()


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion_cls, criterion_reg, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels, suitability in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        suitability = suitability.to(device)

        # Forward pass
        crop_logits, suit_pred = model(features)

        # Calculate losses
        loss_cls = criterion_cls(crop_logits, labels)
        loss_reg = criterion_reg(suit_pred, suitability)

        # Combined loss (weighted)
        loss = 0.7 * loss_cls + 0.3 * loss_reg

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = crop_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(model, val_loader, criterion_cls, criterion_reg, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels, suitability in val_loader:
            features = features.to(device)
            labels = labels.to(device)
            suitability = suitability.to(device)

            # Forward pass
            crop_logits, suit_pred = model(features)

            # Calculate losses
            loss_cls = criterion_cls(crop_logits, labels)
            loss_reg = criterion_reg(suit_pred, suitability)
            loss = 0.7 * loss_cls + 0.3 * loss_reg

            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = crop_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy, all_preds, all_labels


def plot_training_history(history, save_path='../models/training_history.png'):
    """Plot training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy', marker='o')
    ax2.plot(history['val_acc'], label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path='../models/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Crop Recommendation')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")


def train_model(data_path='../data/crop_recommendation_dataset.csv',
                encoders_path='../data/encoders.json',
                model_save_path='../models/crop_recommendation_model.pth',
                epochs=100,
                batch_size=64,
                learning_rate=0.001):
    """Main training function"""

    print("=" * 60)
    print("CROP RECOMMENDATION MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n1. Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df)}")

    # Load encoders
    with open(encoders_path, 'r') as f:
        encoders = json.load(f)
    crop_names = encoders['crop_labels']
    num_crops = len(crop_names)
    print(f"   Crops: {crop_names}")

    # Split dataset (70% train, 15% val, 15% test) - stratified by crop
    print("\n2. Splitting dataset...")
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['crop_label']
    )

    # Second split: val vs test
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['crop_label']
    )

    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")

    # Create datasets
    print("\n3. Creating PyTorch datasets...")
    train_dataset = CropDataset(train_df, fit_scaler=True)
    val_dataset = CropDataset(val_df, scaler=train_dataset.scaler)
    test_dataset = CropDataset(test_df, scaler=train_dataset.scaler)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    print("\n4. Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")

    model = CropRecommendationModel(input_size=9, num_crops=num_crops)
    model = model.to(device)

    # Print model architecture
    print(f"\n   Model Architecture:")
    print(f"   {model}")

    # Loss functions and optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping
    early_stopping = EarlyStopping(patience=15)

    # Training loop
    print(f"\n5. Training model for {epochs} epochs...")
    print("-" * 60)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0
    best_model_state = model.state_dict().copy()  # Initialize with current state

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion_cls, criterion_reg, optimizer, device
        )

        # Validate
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion_cls, criterion_reg, device
        )

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print("-" * 60)

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    print("\n6. Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion_cls, criterion_reg, device
    )

    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.2f}%")

    # Classification report
    print("\n   Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=crop_names))

    # Save model
    print("\n7. Saving model...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': train_dataset.scaler.mean_.tolist(),
        'scaler_scale': train_dataset.scaler.scale_.tolist(),
        'crop_names': crop_names,
        'num_crops': num_crops,
        'input_size': 9,
        'test_accuracy': test_acc,
        'training_history': history,
        'encoders': encoders,
        'trained_at': datetime.now().isoformat()
    }, model_save_path)

    print(f"   Model saved to: {model_save_path}")

    # Plot training history
    print("\n8. Generating plots...")
    plot_training_history(history, save_path='../models/training_history.png')
    plot_confusion_matrix(test_labels, test_preds, crop_names, save_path='../models/confusion_matrix.png')

    print("\n" + "=" * 60)
    print(f"✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)

    return model, history, test_acc


if __name__ == "__main__":
    # Train the model
    model, history, test_acc = train_model(
        epochs=100,
        batch_size=64,
        learning_rate=0.001
    )
