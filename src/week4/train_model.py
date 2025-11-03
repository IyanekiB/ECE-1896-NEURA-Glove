"""
ML Model Trainer
Trains a neural network to map flex sensor angles to joint rotations
Input: 5 flex angles
Output: 10 rotation values (5 joints × 2 axes)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class FlexToRotationDataset(Dataset):
    """PyTorch dataset for flex sensor to rotation mapping"""
    
    def __init__(self, inputs, outputs):
        self.inputs = torch.FloatTensor(inputs)
        self.outputs = torch.FloatTensor(outputs)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class FlexToRotationModel(nn.Module):
    """Neural network for discrete pose mapping
    
    Architecture:
    - Input: 5 flex angles (0-90 degrees)
    - Hidden layers with ReLU activation
    - Output: 10 rotation values (x,y for 5 joints)
    """
    
    def __init__(self, input_dim=5, output_dim=10, hidden_dims=None):
        super(FlexToRotationModel, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 64]
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)
    
    def forward(self, x):
        return self.network(x)


class ModelTrainer:
    """Trainer for flex-to-rotation model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Scalers for normalization
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
    
    def prepare_data(self, aligned_data_file, test_size=0.2, batch_size=32):
        """Load and prepare training data"""
        print(f"\nLoading data from: {aligned_data_file}")
        
        with open(aligned_data_file) as f:
            dataset = json.load(f)
        
        pairs = dataset['pairs']
        print(f"Total training pairs: {len(pairs)}")
        
        # Extract inputs and outputs
        inputs = np.array([p['input'] for p in pairs])
        outputs = np.array([p['output'] for p in pairs])
        
        print(f"\nInput shape: {inputs.shape}")
        print(f"Output shape: {outputs.shape}")
        print(f"\nInput range: [{inputs.min():.2f}, {inputs.max():.2f}]")
        print(f"Output range: [{outputs.min():.2f}, {outputs.max():.2f}]")
        
        # Normalize data
        inputs_scaled = self.input_scaler.fit_transform(inputs)
        outputs_scaled = self.output_scaler.fit_transform(outputs)
        
        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            inputs_scaled, outputs_scaled,
            test_size=test_size,
            random_state=42
        )
        
        print(f"\nTrain samples: {len(X_train)}")
        print(f"Val samples: {len(X_val)}")
        
        # Create dataloaders
        train_dataset = FlexToRotationDataset(X_train, y_train)
        val_dataset = FlexToRotationDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, criterion):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """Train the model"""
        print(f"\n{'='*60}")
        print("TRAINING MODEL")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        print(f"Device: {self.device}")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 20
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if patience_counter >= patience_limit:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plot saved: {save_path}")
        else:
            plt.show()
    
    def save_model(self, model_path, scaler_path):
        """Save trained model and scalers"""
        # Save PyTorch model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_scaler': self.input_scaler,
            'output_scaler': self.output_scaler,
            'history': self.history,
            'model_config': {
                'hidden_dims': getattr(self.model, 'hidden_dims', [64, 128, 64]),
                'input_dim': getattr(self.model.network[0], 'in_features', 5),
                'output_dim': getattr(self.model.network[-1], 'out_features', 10)
            }
        }, model_path)
        
        print(f"\n✓ Model saved: {model_path}")
        print(f"✓ Scalers saved in model checkpoint")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python train_model.py <aligned_data_file>")
        print("\nExample:")
        print("  python train_model.py data/aligned/aligned_session_001_session_001.json")
        sys.exit(1)
    
    aligned_data_file = sys.argv[1]
    
    # Configuration
    INPUT_DIM = 5   # 5 flex sensors
    OUTPUT_DIM = 10  # 5 joints × 2 axes
    HIDDEN_DIMS = [64, 128, 128, 64]
    EPOCHS = 200
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = FlexToRotationModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=HIDDEN_DIMS
    )
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create trainer
    trainer = ModelTrainer(model, device=device)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        aligned_data_file,
        test_size=0.2,
        batch_size=BATCH_SIZE
    )
    
    # Train
    trainer.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Plot and save
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    
    trainer.plot_history(save_path=output_dir / "training_history.png")
    trainer.save_model(
        model_path=output_dir / "flex_to_rotation_model.pth",
        scaler_path=output_dir / "scalers.pkl"
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
