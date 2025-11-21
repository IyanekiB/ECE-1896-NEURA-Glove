"""
NEURA GLOVE - Pose Classifier Training
Train LSTM model for both pose classification and joint prediction
Includes confusion matrix visualization

Usage:
    python train_pose_classifier.py --dataset data/training_dataset.json --epochs 100
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
    SEQUENCE_LENGTH: int = 10  # Use all 10 frames as sequence
    LSTM_HIDDEN_SIZE: int = 128
    LSTM_NUM_LAYERS: int = 2
    LSTM_DROPOUT: float = 0.2
    
    INPUT_SIZE: int = 15   # 5 flex + 4 quat + 3 accel + 3 gyro
    OUTPUT_SIZE: int = 147  # 21 joints × 7
    
    BATCH_SIZE: int = 16  # Smaller batch for pose-based training
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 100
    TRAIN_SPLIT: float = 0.8
    
    # Kalman Filter Parameters
    PROCESS_NOISE: float = 0.01
    MEASUREMENT_NOISE: float = 0.1


# ============================================================================
# DATASET
# ============================================================================

class PoseSequenceDataset(Dataset):
    """Dataset for pose-based training with classification"""
    
    def __init__(self, samples: List[dict], pose_to_idx: Dict[str, int]):
        self.pose_to_idx = pose_to_idx
        self.sequences = []
        self.targets = []
        self.pose_labels = []
        
        # Group samples by pose (each pose has 10 frames)
        poses = {}
        for sample in samples:
            pose_name = sample['pose_name']
            if pose_name not in poses:
                poses[pose_name] = []
            poses[pose_name].append(sample)
        
        for pose_name, pose_samples in poses.items():
            # Sort by frame number
            pose_samples = sorted(pose_samples, key=lambda x: x['frame_number'])
            
            seq_len = 10
            num_sequences = len(pose_samples) // seq_len  # e.g. 300 // 10 = 30
            
            if num_sequences == 0:
                print(f"Warning: Pose '{pose_name}' has too few frames ({len(pose_samples)})")
                continue

            for i in range(num_sequences):
                seq_samples = pose_samples[i*seq_len : (i+1)*seq_len]

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
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            torch.LongTensor([self.pose_labels[idx]])
        )


# ============================================================================
# LSTM MODEL WITH CLASSIFICATION HEAD
# ============================================================================

class PoseClassifierLSTM(nn.Module):
    """LSTM network with dual output: joint prediction + pose classification"""
    
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
            batch_first=True
        )
        
        # Joint prediction head (regression)
        self.joint_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.OUTPUT_SIZE)
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_poses)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        Returns:
            joint_pred: (batch_size, output_size)
            pose_pred: (batch_size, num_poses)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Joint prediction (regression)
        joint_pred = self.joint_head(last_output)
        
        # Pose classification (logits)
        pose_pred = self.classification_head(last_output)
        
        return joint_pred, pose_pred


# ============================================================================
# TRAINER
# ============================================================================

class PoseClassifierTrainer:
    """Trains LSTM model with dual loss: joint regression + pose classification"""
    
    def __init__(self, config: TrainingConfig, pose_names: List[str]):
        self.config = config
        self.pose_names = sorted(pose_names)  # Alphabetical order
        self.pose_to_idx = {name: idx for idx, name in enumerate(self.pose_names)}
        self.idx_to_pose = {idx: name for name, idx in self.pose_to_idx.items()}
        self.num_poses = len(self.pose_names)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = PoseClassifierLSTM(config, self.num_poses).to(self.device)
        
        # Dual loss functions
        self.joint_criterion = nn.MSELoss()
        self.pose_criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
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
        """Load training dataset"""
        print(f"\nLoading dataset from: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        print(f"  ✓ Loaded {len(samples)} samples")
        print(f"    Poses: {data['metadata']['poses']}")
        print(f"    Samples per pose: {data['metadata'].get('samples_per_pose', 'varies')}")
        
        # Create dataset
        dataset = PoseSequenceDataset(samples, self.pose_to_idx)
        
        # Split train/val
        train_size = int(self.config.TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        print(f"\n  Training sequences: {train_size}")
        print(f"  Validation sequences: {val_size}")
        print(f"  Poses to classify: {self.pose_names}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Train for one epoch"""
        self.model.train()
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        correct = 0
        total = 0
        
        for batch_inputs, batch_joint_targets, batch_pose_labels in train_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_joint_targets = batch_joint_targets.to(self.device)
            batch_pose_labels = batch_pose_labels.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            joint_pred, pose_pred = self.model(batch_inputs)
            
            # Dual loss
            joint_loss = self.joint_criterion(joint_pred, batch_joint_targets)
            pose_loss = self.pose_criterion(pose_pred, batch_pose_labels)
            
            # Combined loss (weighted)
            total_loss = joint_loss + 0.5 * pose_loss  # Weight pose loss less
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Accumulate losses
            total_joint_loss += joint_loss.item()
            total_pose_loss += pose_loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(pose_pred.data, 1)
            total += batch_pose_labels.size(0)
            correct += (predicted == batch_pose_labels).sum().item()
        
        avg_joint_loss = total_joint_loss / len(train_loader)
        avg_pose_loss = total_pose_loss / len(train_loader)
        avg_total_loss = avg_joint_loss + 0.5 * avg_pose_loss
        accuracy = 100 * correct / total
        
        return avg_joint_loss, avg_pose_loss, avg_total_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validate model"""
        self.model.eval()
        total_joint_loss = 0.0
        total_pose_loss = 0.0
        correct = 0
        total = 0
        
        # Reset confusion matrix data
        self.val_true_labels = []
        self.val_pred_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_joint_targets, batch_pose_labels in val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_joint_targets = batch_joint_targets.to(self.device)
                batch_pose_labels = batch_pose_labels.squeeze().to(self.device)
                
                joint_pred, pose_pred = self.model(batch_inputs)
                
                joint_loss = self.joint_criterion(joint_pred, batch_joint_targets)
                pose_loss = self.pose_criterion(pose_pred, batch_pose_labels)
                
                total_joint_loss += joint_loss.item()
                total_pose_loss += pose_loss.item()
                
                _, predicted = torch.max(pose_pred.data, 1)
                total += batch_pose_labels.size(0)
                correct += (predicted == batch_pose_labels).sum().item()
                
                # Store for confusion matrix
                self.val_true_labels.extend(batch_pose_labels.cpu().numpy())
                self.val_pred_labels.extend(predicted.cpu().numpy())
        
        avg_joint_loss = total_joint_loss / len(val_loader)
        avg_pose_loss = total_pose_loss / len(val_loader)
        avg_total_loss = avg_joint_loss + 0.5 * avg_pose_loss
        accuracy = 100 * correct / total
        
        return avg_joint_loss, avg_pose_loss, avg_total_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full training loop"""
        print("\n" + "="*60)
        print("TRAINING POSE CLASSIFIER")
        print("="*60)
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Poses: {self.pose_names}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print("="*60)
        print()
        
        best_val_accuracy = 0.0
        patience_counter = 0
        max_patience = 20
        
        for epoch in range(self.config.NUM_EPOCHS):
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
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{self.config.NUM_EPOCHS}] | "
                      f"Joint Loss: {val_joint_loss:.6f} | "
                      f"Pose Loss: {val_pose_loss:.4f} | "
                      f"Accuracy: {val_acc:.2f}% | "
                      f"LR: {current_lr:.2e}")
            
            # Save best model based on accuracy
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
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
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
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
            'pose_names': self.pose_names,
            'pose_to_idx': self.pose_to_idx,
            'history': self.history
        }, filepath)
        
        if filename != 'best_model.pth':
            print(f"✓ Model saved to: {filepath}")
    
    def plot_training_history(self, save_path: str = 'training_history.png'):
        """Plot training history"""
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
        print(f"✓ Training history saved to: {save_path}")
    
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
        print(f"✓ Confusion matrix saved to: {save_path}")
        
        # Print classification report
        print("\nClassification Report:")
        print("="*60)
        report = classification_report(self.val_true_labels, self.val_pred_labels,
                                      target_names=self.pose_names, digits=3)
        print(report)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--batch-size', '-b', type=int, default=16)
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001)
    parser.add_argument('--output', '-o', default='neura_model.pth')
    args = parser.parse_args()
    
    # Load dataset to get pose names
    with open(args.dataset, 'r') as f:
        data = json.load(f)
    
    pose_names = data['metadata']['poses']
    
    # Create config
    config = TrainingConfig()
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    # Create trainer
    trainer = PoseClassifierTrainer(config, pose_names)
    
    # Load dataset
    train_loader, val_loader = trainer.load_dataset(args.dataset)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save final model
    trainer.save_model(args.output)
    
    # Plot results
    trainer.plot_training_history()
    trainer.plot_confusion_matrix()
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"Model: models/{args.output}")
    print(f"Best model: models/best_model.pth")
    print(f"Training plot: training_history.png")
    print(f"Confusion matrix: confusion_matrix.png")
    print("\nNext: Run inference with inference_engine.py")
    print("="*60)


if __name__ == "__main__":
    main()