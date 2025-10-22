"""
NEURA GLOVE - Model Training
Train LSTM + Kalman Filter model on merged training dataset

Usage:
    python train_model.py --dataset training_dataset.json --epochs 100
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration"""
    SEQUENCE_LENGTH: int = 10  # Number of timesteps for LSTM
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    
    INPUT_SIZE: int = 15   # 5 flex + 4 quat + 3 accel + 3 gyro
    OUTPUT_SIZE: int = 147  # 21 joints × 7
    
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100
    TRAIN_SPLIT: float = 0.8
    
    # Kalman Filter Parameters
    PROCESS_NOISE: float = 0.01
    MEASUREMENT_NOISE: float = 0.1


# ============================================================================
# DATASET
# ============================================================================

class HandPoseSequenceDataset(Dataset):
    """Dataset that creates sequences for LSTM training"""
    
    def __init__(self, samples: List[dict], sequence_length: int):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # Create sequences from consecutive frames
        for i in range(len(samples) - sequence_length + 1):
            sequence_samples = samples[i:i + sequence_length]
            
            # Input: sequence of sensor readings (10 frames × 15 features)
            input_sequence = []
            for sample in sequence_samples:
                features = (
                    sample['flex_sensors'] +
                    sample['imu_orientation'] +
                    sample['imu_accel'] +
                    sample['imu_gyro']
                )
                input_sequence.append(features)
            
            # Output: last frame's joint positions and rotations
            last_sample = sequence_samples[-1]
            target = []
            for joint in last_sample['joints']:
                target.extend(joint['position'])  # 3 values
                target.extend(joint['rotation'])  # 4 values
            
            self.sequences.append(np.array(input_sequence, dtype=np.float32))
            self.targets.append(np.array(target, dtype=np.float32))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


# ============================================================================
# LSTM MODEL
# ============================================================================

class TemporalSmoothingLSTM(nn.Module):
    """LSTM network for temporal sequence modeling"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.OUTPUT_SIZE)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            (batch_size, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        output = self.fc(last_output)
        
        return output


# ============================================================================
# TRAINER
# ============================================================================

class ModelTrainer:
    """Trains LSTM model on frame-aligned dataset"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = TemporalSmoothingLSTM(config).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def load_dataset(self, dataset_path: str) -> Tuple[DataLoader, DataLoader]:
        """Load training dataset and create data loaders"""
        print(f"\nLoading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        print(f"  ✓ Loaded {len(samples)} samples")
        print(f"    Input features: {data['metadata']['input_features']}")
        print(f"    Output values: {data['metadata']['output_values']}")
        
        # Create dataset
        dataset = HandPoseSequenceDataset(samples, self.config.SEQUENCE_LENGTH)
        
        # Split train/val
        train_size = int(self.config.TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        print(f"\n  Training sequences: {train_size}")
        print(f"  Validation sequences: {val_size}")
        print(f"  Sequence length: {self.config.SEQUENCE_LENGTH} frames")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs)
            loss = self.criterion(outputs, batch_targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)
        print()
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{self.config.NUM_EPOCHS}] | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print()
        print("="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.6f}")
        print("="*60)
        print()
    
    def save_model(self, filename: str = 'neura_model.pth'):
        """Save model checkpoint"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        filepath = models_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, filepath)
        
        if filename == 'best_model.pth':
            pass  # Silent save during training
        else:
            print(f"✓ Model saved to: {filepath}")
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot and save training history"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rate'], linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {save_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM model on NEURA GLOVE dataset'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Input training dataset JSON file'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    parser.add_argument(
        '--sequence-length', '-seq',
        type=int,
        default=10,
        help='LSTM sequence length (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='neura_model.pth',
        help='Output model filename (default: neura_model.pth)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    config.SEQUENCE_LENGTH = args.sequence_length
    
    # Create trainer
    trainer = ModelTrainer(config)
    
    # Load dataset
    train_loader, val_loader = trainer.load_dataset(args.dataset)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_model(args.output)
    
    # Plot history
    trainer.plot_training_history()
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Model saved to: models/{args.output}")
    print(f"Best model saved to: models/best_model.pth")
    print(f"Training plot: training_history.png")
    print("\nNext step: Run inference with inference_engine.py")
    print("="*60)
    print()


if __name__ == "__main__":
    main()