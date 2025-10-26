"""
NEURA GLOVE - Enhanced Trainer with User Calibration
Trains base model once, then supports user-specific calibration

Usage:
    # Train base model (do this once)
    python enhanced_trainer.py --dataset data/training_dataset.json --epochs 150
    
    # Fine-tune for new user (calibration)
    python enhanced_trainer.py --calibrate --base-model models/base_model.pth \\
                              --user-data data/user_calibration.json --epochs 20
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Training configuration"""
    # Sequence parameters
    SEQUENCE_LENGTH: int = 10  # Use 10-frame sequences
    
    # LSTM architecture
    LSTM_HIDDEN_SIZE: int = 256  # Increased for better capacity
    LSTM_NUM_LAYERS: int = 3     # Deeper network
    LSTM_DROPOUT: float = 0.3
    
    # Input/Output
    INPUT_SIZE: int = 15   # 5 flex + 4 quat + 3 accel + 3 gyro
    OUTPUT_SIZE: int = 147  # 21 joints × 7 (3 pos + 4 rot)
    
    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 150
    TRAIN_SPLIT: float = 0.85
    
    # Calibration hyperparameters (for user-specific fine-tuning)
    CALIBRATION_LEARNING_RATE: float = 0.0001  # Lower LR for fine-tuning
    CALIBRATION_EPOCHS: int = 20


# ============================================================================
# DATASET
# ============================================================================

class HandPoseDataset(Dataset):
    """Dataset for hand pose estimation with temporal sequences"""
    
    def __init__(self, samples: List[dict], pose_to_idx: Dict[str, int], 
                 sequence_length: int = 10):
        self.pose_to_idx = pose_to_idx
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        self.pose_labels = []
        
        # Group samples by pose and dataset
        pose_dataset_groups = {}
        for sample in samples:
            key = (sample['pose_name'], sample.get('dataset_number', 0))
            if key not in pose_dataset_groups:
                pose_dataset_groups[key] = []
            pose_dataset_groups[key].append(sample)
        
        # Create sequences from each group
        for (pose_name, dataset_num), group_samples in pose_dataset_groups.items():
            # Sort by frame number
            group_samples = sorted(group_samples, key=lambda x: x['frame_number'])
            
            # Create overlapping sequences
            num_sequences = len(group_samples) - sequence_length + 1
            
            for i in range(num_sequences):
                seq_samples = group_samples[i:i+sequence_length]
                
                # Input: sequence of sensor readings
                input_sequence = []
                for sample in seq_samples:
                    features = (
                        sample['flex_sensors'] +
                        sample['imu_orientation'] +
                        sample['imu_accel'] +
                        sample['imu_gyro']
                    )
                    input_sequence.append(features)
                
                # Output: last frame's joint data
                target = []
                for joint in seq_samples[-1]['joints']:
                    target.extend(joint['position'])
                    target.extend(joint['rotation'])
                
                self.sequences.append(np.array(input_sequence, dtype=np.float32))
                self.targets.append(np.array(target, dtype=np.float32))
                self.pose_labels.append(self.pose_to_idx[pose_name])
        
        print(f"  Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            torch.LongTensor([self.pose_labels[idx]])
        )


# ============================================================================
# LSTM MODEL
# ============================================================================

class HandPoseLSTM(nn.Module):
    """
    LSTM network for hand pose estimation
    Dual output: joint angles/positions + pose classification
    """
    
    def __init__(self, config: TrainingConfig, num_poses: int):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        
        # Shared LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Joint prediction head (regression)
        self.joint_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, config.OUTPUT_SIZE)
        )
        
        # Pose classification head (auxiliary task)
        self.classification_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_poses)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            joint_pred: (batch_size, output_size)
            pose_pred: (batch_size, num_poses)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Joint prediction
        joint_pred = self.joint_head(last_output)
        
        # Pose classification
        pose_pred = self.classification_head(last_output)
        
        return joint_pred, pose_pred


# ============================================================================
# TRAINER
# ============================================================================

class EnhancedTrainer:
    """Trains base model and supports user calibration"""
    
    def __init__(self, config: TrainingConfig, pose_names: List[str], 
                 calibration_mode: bool = False):
        self.config = config
        self.pose_names = sorted(pose_names)
        self.pose_to_idx = {name: idx for idx, name in enumerate(self.pose_names)}
        self.idx_to_pose = {idx: name for name, idx in self.pose_to_idx.items()}
        self.num_poses = len(self.pose_names)
        self.calibration_mode = calibration_mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        self.model = HandPoseLSTM(config, self.num_poses).to(self.device)
        
        # Loss functions
        self.joint_criterion = nn.MSELoss()
        self.pose_criterion = nn.CrossEntropyLoss()
        
        # Optimizer (different LR for calibration)
        lr = config.CALIBRATION_LEARNING_RATE if calibration_mode else config.LEARNING_RATE
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_joint_loss': [],
            'train_pose_loss': [],
            'train_total_loss': [],
            'val_joint_loss': [],
            'val_pose_loss': [],
            'val_total_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # For confusion matrix
        self.val_true_labels = []
        self.val_pred_labels = []
    
    def load_dataset(self, dataset_path: str) -> Tuple[DataLoader, DataLoader]:
        """Load and split dataset into train/val"""
        print(f"\n📂 Loading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        metadata = data['metadata']
        
        print(f"  Total samples: {len(samples)}")
        print(f"  Poses: {metadata['poses']}")
        if 'pose_statistics' in metadata:
            print(f"  Dataset breakdown:")
            for pose, stats in metadata['pose_statistics'].items():
                print(f"    {pose:15s}: {stats['num_datasets']} datasets, "
                      f"{stats['num_samples']} samples")
        
        # Create dataset
        full_dataset = HandPoseDataset(
            samples, 
            self.pose_to_idx,
            self.config.SEQUENCE_LENGTH
        )
        
        # Split train/val
        train_size = int(self.config.TRAIN_SPLIT * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"  Training sequences: {len(train_dataset)}")
        print(f"  Validation sequences: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        correct = 0
        total = 0
        
        for batch_sequences, batch_targets, batch_pose_labels in train_loader:
            batch_sequences = batch_sequences.to(self.device)
            batch_targets = batch_targets.to(self.device)
            batch_pose_labels = batch_pose_labels.squeeze().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            joint_pred, pose_pred = self.model(batch_sequences)
            
            # Calculate losses
            joint_loss = self.joint_criterion(joint_pred, batch_targets)
            pose_loss = self.pose_criterion(pose_pred, batch_pose_labels)
            
            # Combined loss (joint prediction is primary task)
            total_loss = joint_loss + 0.3 * pose_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_joint_loss += joint_loss.item()
            total_pose_loss += pose_loss.item()
            
            _, predicted = torch.max(pose_pred, 1)
            total += batch_pose_labels.size(0)
            correct += (predicted == batch_pose_labels).sum().item()
        
        avg_joint_loss = total_joint_loss / len(train_loader)
        avg_pose_loss = total_pose_loss / len(train_loader)
        avg_total_loss = avg_joint_loss + 0.3 * avg_pose_loss
        accuracy = 100 * correct / total
        
        return avg_joint_loss, avg_pose_loss, avg_total_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validate model"""
        self.model.eval()
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        correct = 0
        total = 0
        
        self.val_true_labels = []
        self.val_pred_labels = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets, batch_pose_labels in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                batch_pose_labels = batch_pose_labels.squeeze().to(self.device)
                
                # Forward pass
                joint_pred, pose_pred = self.model(batch_sequences)
                
                # Calculate losses
                joint_loss = self.joint_criterion(joint_pred, batch_targets)
                pose_loss = self.pose_criterion(pose_pred, batch_pose_labels)
                
                total_joint_loss += joint_loss.item()
                total_pose_loss += pose_loss.item()
                
                _, predicted = torch.max(pose_pred, 1)
                total += batch_pose_labels.size(0)
                correct += (predicted == batch_pose_labels).sum().item()
                
                # Store for confusion matrix
                self.val_true_labels.extend(batch_pose_labels.cpu().numpy())
                self.val_pred_labels.extend(predicted.cpu().numpy())
        
        avg_joint_loss = total_joint_loss / len(val_loader)
        avg_pose_loss = total_pose_loss / len(val_loader)
        avg_total_loss = avg_joint_loss + 0.3 * avg_pose_loss
        accuracy = 100 * correct / total
        
        return avg_joint_loss, avg_pose_loss, avg_total_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.CALIBRATION_EPOCHS if self.calibration_mode else self.config.NUM_EPOCHS
        
        mode_str = "CALIBRATION" if self.calibration_mode else "BASE MODEL TRAINING"
        
        print("\n" + "="*70)
        print(f"{mode_str}")
        print("="*70)
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Poses: {self.pose_names}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*70)
        print()
        
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = 25 if not self.calibration_mode else 10
        
        for epoch in range(num_epochs):
            # Train
            train_joint_loss, train_pose_loss, train_total_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_joint_loss, val_pose_loss, val_total_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_total_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_joint_loss'].append(train_joint_loss)
            self.history['train_pose_loss'].append(train_pose_loss)
            self.history['train_total_loss'].append(train_total_loss)
            self.history['val_joint_loss'].append(val_joint_loss)
            self.history['val_pose_loss'].append(val_pose_loss)
            self.history['val_total_loss'].append(val_total_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                      f"Joint: {val_joint_loss:.6f} | "
                      f"Pose: {val_pose_loss:.4f} | "
                      f"Acc: {val_acc:.2f}% | "
                      f"LR: {current_lr:.2e}")
            
            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                best_val_accuracy = val_acc
                self.save_model('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break
        
        print()
        print("="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Best validation accuracy: {best_val_accuracy:.2f}%")
        print("="*70)
        print()
    
    def load_base_model(self, model_path: str):
        """Load pretrained base model for calibration"""
        print(f"\n📥 Loading base model: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  ✓ Base model loaded successfully")
        print(f"  Original training: {len(checkpoint.get('history', {}).get('val_accuracy', []))} epochs")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        filepath = models_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'pose_names': self.pose_names,
            'pose_to_idx': self.pose_to_idx,
            'history': self.history,
            'calibration_mode': self.calibration_mode
        }, filepath)
        
        if filename != 'best_model.pth':
            print(f"  💾 Model saved: {filepath}")
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Joint loss
        axes[0, 0].plot(self.history['train_joint_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_joint_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].set_title('Joint Prediction Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Pose loss
        axes[0, 1].plot(self.history['train_pose_loss'], label='Train', linewidth=2, color='orange')
        axes[0, 1].plot(self.history['val_pose_loss'], label='Validation', linewidth=2, color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Cross-Entropy Loss')
        axes[0, 1].set_title('Pose Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1, 0].plot(self.history['val_accuracy'], linewidth=2, color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Validation Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'], linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Training history: {save_path}")
    
    def plot_confusion_matrix(self, save_path: str = 'confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(self.val_true_labels, self.val_pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.pose_names,
                    yticklabels=self.pose_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Pose Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Pose', fontsize=12)
        plt.xlabel('Predicted Pose', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Confusion matrix: {save_path}")
        
        # Print classification report
        print("\n📋 Classification Report:")
        print("="*70)
        report = classification_report(self.val_true_labels, self.val_pred_labels,
                                      target_names=self.pose_names, digits=3)
        print(report)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train NEURA Glove hand pose model')
    
    # Mode selection
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibration mode (fine-tune for new user)')
    
    # Dataset
    parser.add_argument('--dataset', '-d', required=True,
                       help='Training dataset JSON file')
    
    # Base model (for calibration)
    parser.add_argument('--base-model', '-b',
                       help='Base model to fine-tune (for calibration mode)')
    
    # Training parameters
    parser.add_argument('--epochs', '-e', type=int,
                       help='Number of epochs (default: 150 for base, 20 for calibration)')
    parser.add_argument('--batch-size', '-bs', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', '-lr', type=float,
                       help='Learning rate (default: 0.001 for base, 0.0001 for calibration)')
    
    # Output
    parser.add_argument('--output', '-o', default='base_model.pth',
                       help='Output model filename (default: base_model.pth)')
    
    args = parser.parse_args()
    
    # Load dataset to get pose names
    with open(args.dataset, 'r') as f:
        data = json.load(f)
    
    pose_names = data['metadata']['poses']
    
    # Create config
    config = TrainingConfig()
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    # Create trainer
    trainer = EnhancedTrainer(config, pose_names, calibration_mode=args.calibrate)
    
    # Update learning rate if specified
    if args.learning_rate:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = args.learning_rate
    
    # Load base model if calibration mode
    if args.calibrate:
        if not args.base_model:
            print("✗ Error: --base-model required for calibration mode")
            return
        trainer.load_base_model(args.base_model)
    
    # Load dataset
    train_loader, val_loader = trainer.load_dataset(args.dataset)
    
    # Train
    trainer.train(train_loader, val_loader, args.epochs)
    
    # Save final model
    trainer.save_model(args.output)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix()
    
    print("\n" + "="*70)
    print("✅ SUCCESS!")
    print("="*70)
    print(f"  Final model: models/{args.output}")
    print(f"  Best model: models/best_model.pth")
    print(f"  Training history: training_history.png")
    print(f"  Confusion matrix: confusion_matrix.png")
    
    if not args.calibrate:
        print(f"\n📋 Next steps:")
        print(f"   1. Test inference:")
        print(f"      python inference_engine.py --model models/best_model.pth")
        print(f"   2. For new users, collect calibration data and run:")
        print(f"      python enhanced_trainer.py --calibrate \\")
        print(f"             --base-model models/best_model.pth \\")
        print(f"             --dataset data/user_calibration.json --epochs 20")
    else:
        print(f"\n📋 Next step:")
        print(f"   Run inference with calibrated model:")
        print(f"   python inference_engine.py --model models/{args.output}")
    
    print("="*70)


if __name__ == "__main__":
    main()