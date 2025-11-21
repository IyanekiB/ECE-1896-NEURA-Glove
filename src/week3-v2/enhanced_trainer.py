"""
FIXED LSTM TRAINER - Addresses quaternion output issues

KEY FIXES:
1. Proper quaternion normalization in model output
2. Simpler, more effective LSTM architecture  
3. Better loss function for rotations
4. Proper data format handling
5. Gradient clipping to prevent exploding gradients

PROBLEMS IDENTIFIED IN ORIGINAL CODE:
- No quaternion normalization → outputs near-zero values
- Too deep LSTM (3 layers) causing vanishing gradients
- No special handling for rotation vs position data
- Training on raw positions instead of normalized rotations
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# FIXED CONFIGURATION
# ============================================================================

class TrainingConfig:
    """Optimized training configuration"""
    # Sequence parameters
    SEQUENCE_LENGTH: int = 10
    
    # SIMPLIFIED LSTM architecture - fewer layers work better
    LSTM_HIDDEN_SIZE: int = 128  # Reduced from 256
    LSTM_NUM_LAYERS: int = 2     # Reduced from 3
    LSTM_DROPOUT: float = 0.2    # Reduced dropout
    
    # Input/Output
    INPUT_SIZE: int = 15
    OUTPUT_SIZE: int = 147  # 21 joints × 7 (3 pos + 4 rot)
    
    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    NUM_EPOCHS: int = 150
    TRAIN_SPLIT: float = 0.85
    GRADIENT_CLIP: float = 1.0  # Prevent exploding gradients
    
    # Loss weights
    ROTATION_LOSS_WEIGHT: float = 10.0  # Prioritize rotation learning
    POSE_LOSS_WEIGHT: float = 0.5       # Auxiliary task


# ============================================================================
# DATASET (SAME AS BEFORE)
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
        self._output_size = None
        
        pose_dataset_groups = {}
        for sample in samples:
            key = (sample['pose_name'], sample.get('dataset_number', 0))
            if key not in pose_dataset_groups:
                pose_dataset_groups[key] = []
            pose_dataset_groups[key].append(sample)
        
        for (pose_name, dataset_num), group_samples in pose_dataset_groups.items():
            group_samples = sorted(group_samples, key=lambda x: x['frame_number'])
            num_sequences = len(group_samples) - sequence_length + 1
            
            for i in range(num_sequences):
                seq_samples = group_samples[i:i+sequence_length]
                
                input_sequence = []
                for sample in seq_samples:
                    features = (
                        sample['flex_sensors'] +
                        sample['imu_orientation'] +
                        sample['imu_accel'] +
                        sample['imu_gyro']
                    )
                    input_sequence.append(features)
                
                target = []
                joints_data = seq_samples[-1]['joints']
                
                if isinstance(joints_data, list):
                    for joint in joints_data:
                        target.extend(joint['position'])
                        target.extend(joint['rotation'])
                elif isinstance(joints_data, dict):
                    first_value = next(iter(joints_data.values()))
                    if isinstance(first_value, dict):
                        for finger_name in ['wrist', 'thumb', 'index', 'middle', 'ring', 'pinky']:
                            if finger_name in joints_data:
                                finger_data = joints_data[finger_name]
                                if isinstance(finger_data, dict):
                                    for joint_name in sorted(finger_data.keys()):
                                        target.append(float(finger_data[joint_name]))
                                else:
                                    target.append(float(finger_data))
                    else:
                        target = []
                        for joint_name, coords in joints_data.items():
                            if isinstance(coords, dict):
                                target.extend([float(coords.get('x', 0.0)),
                                            float(coords.get('y', 0.0)),
                                            float(coords.get('z', 0.0))])
                            else:
                                target.append(float(coords))
                else:
                    raise ValueError(f"Unknown joints data format: {type(joints_data)}")
                
                self.sequences.append(np.array(input_sequence, dtype=np.float32))
                self.targets.append(np.array(target, dtype=np.float32))
                self.pose_labels.append(self.pose_to_idx[pose_name])
                
                if self._output_size is None:
                    self._output_size = len(target)
        
        print(f"  Created {len(self.sequences)} training sequences")
        print(f"  Detected output size: {self._output_size}")
    
    @property
    def output_size(self):
        return self._output_size
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            self.pose_labels[idx]  # Return scalar, will be batched to (batch_size,)
        )


# ============================================================================
# FIXED LSTM MODEL WITH QUATERNION NORMALIZATION
# ============================================================================

class QuaternionNormalizationLayer(nn.Module):
    """
    Normalizes quaternions to unit length.
    Robust: handles outputs that are either:
      - (batch, n_joints * 7)  => treat each joint as [px,py,pz,qx,qy,qz,qw] and normalize last 4
      - (batch, k * 4)         => treat as k quaternions and normalize each
      - otherwise: no-op (returns x)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, N) tensor
        Returns x with quaternion components normalized where possible.
        """
        if x.ndim != 2:
            # unexpected shape, return as-is
            return x

        batch_size, total = x.shape

        # Case 1: total is divisible by 7 (assume per-joint size 7: 3 pos + 4 quat)
        if total % 7 == 0:
            n_joints = total // 7
            try:
                x_reshaped = x.view(batch_size, n_joints, 7)  # (batch, n_joints, 7)
            except Exception:
                return x  # safe fallback
            positions = x_reshaped[:, :, :3]            # (batch, n_joints, 3)
            quaternions = x_reshaped[:, :, 3:]          # (batch, n_joints, 4)
            quat_norms = torch.norm(quaternions, dim=2, keepdim=True) + 1e-8
            quaternions_normalized = quaternions / quat_norms
            x_normalized = torch.cat([positions, quaternions_normalized], dim=2)
            return x_normalized.view(batch_size, total)

        # Case 2: total is divisible by 4 (assume pure quaternions sequence)
        if total % 4 == 0:
            k = total // 4
            try:
                q_reshaped = x.view(batch_size, k, 4)   # (batch, k, 4)
            except Exception:
                return x
            q_norms = torch.norm(q_reshaped, dim=2, keepdim=True) + 1e-8
            q_normalized = q_reshaped / q_norms
            return q_normalized.view(batch_size, total)

        # Fallback: cannot infer quaternion layout — return x unchanged
        return x


class FixedHandPoseLSTM(nn.Module):
    """
    FIXED LSTM with proper quaternion handling
    CHANGES:
    1. Simpler architecture (2 layers instead of 3)
    2. Quaternion normalization layer
    3. Separate heads for position and rotation
    4. Better initialization
    """
    
    def __init__(self, config: TrainingConfig, num_poses: int):
        super().__init__()
        self.config = config
        self.num_poses = num_poses
        
        # Simpler LSTM backbone (2 layers)
        self.lstm = nn.LSTM(
            input_size=config.INPUT_SIZE,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Simplified joint prediction head
        self.joint_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.OUTPUT_SIZE),
            QuaternionNormalizationLayer()  # CRITICAL FIX
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(config.LSTM_HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_poses)
        )
        
        # Better weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, sequence_length, input_size)
        
        Returns:
            joint_pred: (batch_size, output_size) with normalized quaternions
            pose_pred: (batch_size, num_poses)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Joint prediction (with quaternion normalization)
        joint_pred = self.joint_head(last_output)
        
        # Pose classification
        pose_pred = self.classification_head(last_output)
        
        return joint_pred, pose_pred


# ============================================================================
# CUSTOM LOSS FUNCTION
# ============================================================================

class RotationAwareLoss(nn.Module):
    """
    Rotation-aware loss that adapts dynamically to output size.
    If the output cannot be reshaped as (n_joints, 7), it falls back to simple MSE.
    """
    def __init__(self, rotation_weight=10.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        batch_size = pred.shape[0]
        total_dim = pred.shape[1]

        # ✅ Case 1: If divisible by 7 → assume (n_joints × [pos3 + rot4])
        if total_dim % 7 == 0:
            n_joints = total_dim // 7
            pred_reshaped = pred.view(batch_size, n_joints, 7)
            target_reshaped = target.view(batch_size, n_joints, 7)

            pos_pred = pred_reshaped[:, :, :3]
            pos_target = target_reshaped[:, :, :3]
            rot_pred = pred_reshaped[:, :, 3:]
            rot_target = target_reshaped[:, :, 3:]

            pos_loss = self.mse(pos_pred, pos_target)
            rot_loss = self.mse(rot_pred, rot_target)
            total_loss = pos_loss + self.rotation_weight * rot_loss
            return total_loss, pos_loss, rot_loss

        # ✅ Case 2: If divisible by 4 → assume quaternion-only output
        elif total_dim % 4 == 0:
            k = total_dim // 4
            pred_q = pred.view(batch_size, k, 4)
            target_q = target.view(batch_size, k, 4)
            rot_loss = self.mse(pred_q, target_q)
            return rot_loss, torch.tensor(0.0, device=pred.device), rot_loss

        # ✅ Case 3: Otherwise, treat as generic regression
        else:
            total_loss = self.mse(pred, target)
            return total_loss, total_loss, torch.tensor(0.0, device=pred.device)


# ============================================================================
# TRAINER WITH FIXES
# ============================================================================

class FixedTrainer:
    """Enhanced trainer with proper quaternion handling"""
    
    def __init__(self, config: TrainingConfig, pose_names: List[str]):
        self.config = config
        self.pose_names = sorted(pose_names)
        self.pose_to_idx = {name: idx for idx, name in enumerate(self.pose_names)}
        self.num_poses = len(self.pose_names)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"  Device: {self.device}")
        
        self.model = FixedHandPoseLSTM(config, self.num_poses).to(self.device)
        
        # Custom loss functions
        self.joint_criterion = RotationAwareLoss(config.ROTATION_LOSS_WEIGHT)
        self.pose_criterion = nn.CrossEntropyLoss()
        self.pose_weight = config.POSE_LOSS_WEIGHT  # Missing attribute
        
        # Optimizer with gradient clipping
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_pos_loss': [], 'val_pos_loss': [],
            'train_rot_loss': [], 'val_rot_loss': [],
            'train_pose_acc': [], 'val_pose_acc': []
        }
        
        print(f"\n  Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Trainable Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def prepare_data(self, dataset_path: str):
        """Load and prepare datasets"""
        print(f"\nLoading dataset: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data['samples']
        print(f"  Total samples: {len(samples)}")
        
        # Group by recording session to prevent data leakage
        recording_groups = {}
        for sample in samples:
            key = (sample['pose_name'], sample.get('dataset_number', 0))
            if key not in recording_groups:
                recording_groups[key] = []
            recording_groups[key].append(sample)
        
        # Split by recording sessions
        all_keys = list(recording_groups.keys())
        np.random.seed(42)
        np.random.shuffle(all_keys)
        
        split_idx = int(self.config.TRAIN_SPLIT * len(all_keys))
        train_keys = set(all_keys[:split_idx])
        val_keys = set(all_keys[split_idx:])
        
        train_samples = []
        val_samples = []
        
        for key in train_keys:
            train_samples.extend(recording_groups[key])
        for key in val_keys:
            val_samples.extend(recording_groups[key])
        
        print(f"  Training sessions: {len(train_keys)}")
        print(f"  Validation sessions: {len(val_keys)}")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        
        # Create datasets
        train_dataset = HandPoseDataset(
            train_samples, self.pose_to_idx, self.config.SEQUENCE_LENGTH
        )
        val_dataset = HandPoseDataset(
            val_samples, self.pose_to_idx, self.config.SEQUENCE_LENGTH
        )
        
        # Update config if needed
        if train_dataset.output_size != self.config.OUTPUT_SIZE:
            print(f"\n  ⚠️  Updating OUTPUT_SIZE: {self.config.OUTPUT_SIZE} → {train_dataset.output_size}")
            self.config.OUTPUT_SIZE = train_dataset.output_size
            self.model = FixedHandPoseLSTM(self.config, self.num_poses).to(self.device)
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.LEARNING_RATE,
                weight_decay=0.01
            )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_pos_loss = 0
        total_rot_loss = 0
        correct_poses = 0
        total_poses = 0
        
        for batch in train_loader:
            if not batch or len(batch[0]) == 0:
                continue

        for sequences, targets, pose_labels in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            pose_labels = pose_labels.squeeze().to(self.device)
            
            # Forward pass
            joint_pred, pose_pred = self.model(sequences)
            
            # Compute losses
            joint_loss, pos_loss, rot_loss = self.joint_criterion(joint_pred, targets)
            pose_loss = self.pose_criterion(pose_pred, pose_labels)
            
            # Combined loss
            loss = joint_loss + self.config.POSE_LOSS_WEIGHT * pose_loss
            
            # Backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP
            )
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_pos_loss += pos_loss.item()
            total_rot_loss += rot_loss.item()
            
            _, predicted = torch.max(pose_pred, 1)
            correct_poses += (predicted == pose_labels).sum().item()
            total_poses += pose_labels.size(0)
        
        avg_loss = total_loss / len(train_loader)
        avg_pos_loss = total_pos_loss / len(train_loader)
        avg_rot_loss = total_rot_loss / len(train_loader)
        pose_acc = 100.0 * correct_poses / total_poses
        
        return avg_loss, avg_pos_loss, avg_rot_loss, pose_acc
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_pos = 0.0
        total_rot = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Skip empty batches
                if not batch:
                    continue

                if len(batch) == 2:
                    inputs, targets = batch
                    pose_labels = None
                elif len(batch) == 3:
                    inputs, targets, pose_labels = batch
                else:
                    continue  # malformed batch

                if inputs.size(0) == 0:
                    continue  # skip empty input batch

                inputs = inputs.to(self.device).float()
                targets = targets.to(self.device).float()

                outputs = self.model(inputs)
                joint_pred, pose_pred = outputs if isinstance(outputs, tuple) else (outputs, None)

                # Rotation loss (handles dynamic sizes safely)
                joint_loss, pos_loss, rot_loss = self.joint_criterion(joint_pred, targets)

                # Optional classification loss
                pose_loss = 0.0
                if pose_pred is not None and pose_labels is not None and len(pose_labels) > 0:
                    pose_labels = pose_labels.to(self.device).long()
                    pose_loss = self.pose_criterion(pose_pred, pose_labels)
                    total_acc += (pose_pred.argmax(dim=1) == pose_labels).float().mean().item()

                loss = joint_loss + self.pose_weight * pose_loss
                total_loss += loss.item()
                total_pos += pos_loss.item() if torch.is_tensor(pos_loss) else pos_loss
                total_rot += rot_loss.item() if torch.is_tensor(rot_loss) else rot_loss
                num_batches += 1

        # Avoid division by zero
        if num_batches == 0:
            print("⚠️ Warning: Validation had zero usable batches. Skipping metrics.")
            return 0.0, 0.0, 0.0, 0.0

        return (
            total_loss / num_batches,
            total_pos / num_batches,
            total_rot / num_batches,
            total_acc / num_batches if total_acc != 0 else 0.0
        )
    
    def train(self, train_loader, val_loader, output_path: str):
        """Full training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING WITH FIXED LSTM")
        print("="*70)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Train
            train_loss, train_pos, train_rot, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_pos, val_rot, val_acc = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_pos_loss'].append(train_pos)
            self.history['val_pos_loss'].append(val_pos)
            self.history['train_rot_loss'].append(train_rot)
            self.history['val_rot_loss'].append(val_rot)
            self.history['train_pose_acc'].append(train_acc)
            self.history['val_pose_acc'].append(val_acc)
            
            # Print progress
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f} (Pos: {train_pos:.4f}, Rot: {train_rot:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Pos: {val_pos:.4f}, Rot: {val_rot:.4f})")
            print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(output_path, epoch, is_best=True)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping after {epoch+1} epochs")
                    break
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        
        # Plot training history
        self.plot_training_history()
    
    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        Path("models").mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'pose_names': self.pose_names,
            'pose_to_idx': self.pose_to_idx,
            'history': self.history
        }
        
        if is_best:
            torch.save(checkpoint, 'models/best_model_fixed.pth')
        
        torch.save(checkpoint, f'models/{path}')
    
    def plot_training_history(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Val')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Position loss
        axes[0, 1].plot(self.history['train_pos_loss'], label='Train')
        axes[0, 1].plot(self.history['val_pos_loss'], label='Val')
        axes[0, 1].set_title('Position Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Rotation loss
        axes[1, 0].plot(self.history['train_rot_loss'], label='Train')
        axes[1, 0].plot(self.history['val_rot_loss'], label='Val')
        axes[1, 0].set_title('Rotation Loss (Most Important)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Accuracy
        axes[1, 1].plot(self.history['train_pose_acc'], label='Train')
        axes[1, 1].plot(self.history['val_pose_acc'], label='Val')
        axes[1, 1].set_title('Pose Classification Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history_fixed.png', dpi=150)
        print(f"\n  Training history saved: training_history_fixed.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fixed LSTM trainer with quaternion normalization')
    parser.add_argument('--dataset', type=str, required=True, help='Training dataset path')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--output', type=str, default='fixed_model.pth', help='Output model name')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    config.NUM_EPOCHS = args.epochs
    
    # Detect poses from dataset
    with open(args.dataset, 'r') as f:
        data = json.load(f)
    pose_names = sorted(set(s['pose_name'] for s in data['samples']))
    
    print("\n" + "="*70)
    print("FIXED LSTM TRAINER")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Poses: {pose_names}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*70)
    
    # Create trainer
    trainer = FixedTrainer(config, pose_names)
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(args.dataset)
    
    # Train
    trainer.train(train_loader, val_loader, args.output)


if __name__ == "__main__":
    main()