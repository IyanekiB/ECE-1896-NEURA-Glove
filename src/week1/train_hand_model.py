import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path


class HandPoseDataset(Dataset):
    """Dataset for sensor-to-joints mapping"""
    
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract input features (15 values)
        sensors = sample['sensors']
        input_features = (
            sensors['flex'] +  # 5 values
            sensors['imu_orientation'] +  # 4 values
            sensors['imu_accel'] +  # 3 values
            sensors['imu_gyro']  # 3 values
        )
        
        # Extract output values (147 values: 21 joints × 7)
        joints = sample['ground_truth']['joints']
        output_values = []
        for joint in joints:
            output_values.extend(joint['position'])  # 3 values
            output_values.extend(joint['rotation'])  # 4 values
        
        return (
            torch.FloatTensor(input_features),
            torch.FloatTensor(output_values)
        )


class HandPoseModel(nn.Module):
    """
    Simple feedforward neural network
    Input: 15 features (5 flex + 10 IMU)
    Output: 147 values (21 joints × 7)
    """
    
    def __init__(self, input_size=15, output_size=147):
        super(HandPoseModel, self).__init__()
        
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(256, output_size)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(dataset_path, epochs=100, batch_size=32, learning_rate=0.001):
    """Train the hand pose estimation model"""
    
    print("="*60)
    print("HAND POSE MODEL TRAINING")
    print("="*60)
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    samples = data['samples']
    total_samples = len(samples)
    
    print(f"Loaded {total_samples} samples")
    print(f"  Input features: {data['metadata']['input_features']}")
    print(f"  Output values: {data['metadata']['output_values']}")
    
    # Split into train/validation (80/20)
    split_idx = int(0.8 * total_samples)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    
    # Create datasets and dataloaders
    train_dataset = HandPoseDataset(train_samples)
    val_dataset = HandPoseDataset(val_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = HandPoseModel().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_hand_model.pth')
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {current_lr:.6f}")
            print(f"  Best Val:   {best_val_loss:.6f}")
            print()
    
    print("="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)
    
    # Plot training curves
    plot_training_history(history)
    
    # Evaluate on validation set
    evaluate_model(model, val_loader, device)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation loss"""
    
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning rate plot
    plt.subplot(1, 2, 2)
    plt.plot(history['learning_rate'], linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("\nTraining curves saved to training_history.png")


def evaluate_model(model, val_loader, device):
    """Detailed evaluation on validation set"""
    
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs = batch_inputs.to(device)
            outputs = model(batch_inputs)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # Overall metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    print(f"\nOverall Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Per-joint analysis
    print(f"\nPer-Joint Analysis:")
    print(f"  {'Joint':<10} {'Position Error':<20} {'Rotation Error'}")
    print(f"  {'-'*10} {'-'*20} {'-'*20}")
    
    for joint_id in range(21):
        # Each joint has 7 values (3 pos + 4 rot)
        start_idx = joint_id * 7
        
        # Position error (first 3 values)
        pos_pred = predictions[:, start_idx:start_idx+3]
        pos_true = targets[:, start_idx:start_idx+3]
        pos_error = np.mean(np.linalg.norm(pos_pred - pos_true, axis=1))
        
        # Rotation error (next 4 values)
        rot_pred = predictions[:, start_idx+3:start_idx+7]
        rot_true = targets[:, start_idx+3:start_idx+7]
        rot_error = np.mean(np.linalg.norm(rot_pred - rot_true, axis=1))
        
        joint_name = get_joint_name(joint_id)
        print(f"  {joint_name:<10} {pos_error:<20.6f} {rot_error:.6f}")
    
    print("\n" + "="*60)


def get_joint_name(joint_id):
    """Get human-readable joint name"""
    joint_names = [
        "Wrist", "Thumb1", "Thumb2", "Thumb3", "ThumbTip",
        "Index1", "Index2", "Index3", "IndexTip",
        "Middle1", "Middle2", "Middle3", "MiddleTip",
        "Ring1", "Ring2", "Ring3", "RingTip",
        "Pinky1", "Pinky2", "Pinky3", "PinkyTip"
    ]
    return joint_names[joint_id]


if __name__ == "__main__":
    import sys
    
    dataset_path = "datasets/training_dataset.json"
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found: {dataset_path}")
        print("\nFirst run: python create_training_dataset.py")
        sys.exit(1)
    
    # Train model
    model, history = train_model(
        dataset_path,
        epochs=100,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\nModel saved to: best_hand_model.pth")
    print("Ready for camera-free inference!")