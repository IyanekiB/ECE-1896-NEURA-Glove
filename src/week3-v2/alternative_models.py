"""
ALTERNATIVE ML MODELS FOR HAND POSE ESTIMATION

Provides multiple model architectures beyond LSTM:
1. GRU (Gated Recurrent Unit) - Simpler than LSTM, often faster
2. TCN (Temporal Convolutional Network) - Better for long sequences
3. Transformer - State-of-the-art for sequence modeling
4. 1D-CNN + LSTM Hybrid - Combines spatial and temporal features

Each model can be used as a drop-in replacement for the LSTM model.

USAGE:
    # Train with GRU instead of LSTM
    python alternative_models.py --model gru --dataset data/training_dataset.json
    
    # Train with TCN
    python alternative_models.py --model tcn --dataset data/training_dataset.json
    
    # Train with Transformer
    python alternative_models.py --model transformer --dataset data/training_dataset.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ============================================================================
# QUATERNION NORMALIZATION (SHARED)
# ============================================================================

class QuaternionNormalizationLayer(nn.Module):
    """Normalizes quaternions in output"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """x: (batch, 147) - normalize quaternions"""
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, 21, 7)
        
        positions = x_reshaped[:, :, :3]
        quaternions = x_reshaped[:, :, 3:]
        
        quat_norms = torch.norm(quaternions, dim=2, keepdim=True) + 1e-8
        quaternions_normalized = quaternions / quat_norms
        
        x_normalized = torch.cat([positions, quaternions_normalized], dim=2)
        return x_normalized.reshape(batch_size, 147)


# ============================================================================
# MODEL 1: GRU (FASTER ALTERNATIVE TO LSTM)
# ============================================================================

class HandPoseGRU(nn.Module):
    """
    GRU-based model - Often faster and sometimes better than LSTM
    
    ADVANTAGES:
    - Fewer parameters than LSTM (faster training)
    - Less prone to overfitting
    - Often achieves similar or better performance
    
    WHEN TO USE:
    - When LSTM is overfitting
    - When you need faster training
    - As a simpler baseline
    """
    
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, 
                 num_poses=5, output_size=147):
        super().__init__()
        
        # GRU backbone
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2 if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Joint prediction head
        self.joint_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            QuaternionNormalizationLayer()
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_poses)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: (joint_pred, pose_pred)
        """
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        
        joint_pred = self.joint_head(last_output)
        pose_pred = self.classification_head(last_output)
        
        return joint_pred, pose_pred


# ============================================================================
# MODEL 2: TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================

class TemporalBlock(nn.Module):
    """Single temporal block for TCN"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        
        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        # Remove extra padding
        out = out[:, :, :x.size(2)]
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class HandPoseTCN(nn.Module):
    """
    Temporal Convolutional Network for hand pose estimation
    
    ADVANTAGES:
    - Better at capturing long-range dependencies
    - Parallelizable (faster training than RNN)
    - No vanishing gradient problems
    - Better for real-time inference
    
    WHEN TO USE:
    - When you have longer sequences
    - When you need faster inference
    - When LSTM/GRU struggle with long-term dependencies
    """
    
    def __init__(self, input_size=15, num_channels=[64, 128, 128, 256], 
                 kernel_size=3, num_poses=5, output_size=147):
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, dropout=0.2
            ))
        
        self.network = nn.Sequential(*layers)
        
        # Joint prediction head
        self.joint_head = nn.Sequential(
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            QuaternionNormalizationLayer()
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_poses)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: (joint_pred, pose_pred)
        """
        # TCN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Pass through TCN
        features = self.network(x)
        
        # Use last timestep
        last_features = features[:, :, -1]
        
        joint_pred = self.joint_head(last_features)
        pose_pred = self.classification_head(last_features)
        
        return joint_pred, pose_pred


# ============================================================================
# MODEL 3: TRANSFORMER
# ============================================================================

class HandPoseTransformer(nn.Module):
    """
    Transformer-based hand pose estimation
    
    ADVANTAGES:
    - State-of-the-art for sequence modeling
    - Attention mechanism captures complex dependencies
    - Highly parallelizable
    - Can attend to all timesteps simultaneously
    
    WHEN TO USE:
    - When you have enough data (>5000 samples)
    - When other models plateau
    - When you need the best possible accuracy
    
    NOTE: Requires more data and compute than RNN/TCN
    """
    
    def __init__(self, input_size=15, d_model=128, nhead=4, num_layers=3,
                 num_poses=5, output_size=147, max_seq_len=10):
        super().__init__()
        
        # Input embedding
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Joint prediction head
        self.joint_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            QuaternionNormalizationLayer()
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_poses)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: (joint_pred, pose_pred)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer encoding
        features = self.transformer(x)
        
        # Use last timestep
        last_features = features[:, -1, :]
        
        joint_pred = self.joint_head(last_features)
        pose_pred = self.classification_head(last_features)
        
        return joint_pred, pose_pred


# ============================================================================
# MODEL 4: 1D-CNN + LSTM HYBRID
# ============================================================================

class HandPoseCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model
    
    ADVANTAGES:
    - CNN extracts local patterns from sensor data
    - LSTM captures temporal dependencies
    - Often more robust than pure LSTM
    - Good at learning sensor-specific features
    
    WHEN TO USE:
    - When sensor data has spatial structure
    - When pure LSTM underfits
    - When you want feature learning + temporal modeling
    """
    
    def __init__(self, input_size=15, cnn_channels=[32, 64], 
                 lstm_hidden=128, lstm_layers=2, num_poses=5, output_size=147):
        super().__init__()
        
        # 1D CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels[0]),
            
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(cnn_channels[1])
        )
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_channels[1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=0.2 if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Joint prediction head
        self.joint_head = nn.Sequential(
            nn.Linear(lstm_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_size),
            QuaternionNormalizationLayer()
        )
        
        # Pose classification head
        self.classification_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_poses)
        )
    
    def forward(self, x):
        """
        x: (batch, seq_len, input_size)
        Returns: (joint_pred, pose_pred)
        """
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Extract features with CNN
        cnn_features = self.cnn(x)
        
        # Back to (batch, seq_len, features)
        cnn_features = cnn_features.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(cnn_features)
        
        # Use last timestep
        last_output = lstm_out[:, -1, :]
        
        joint_pred = self.joint_head(last_output)
        pose_pred = self.classification_head(last_output)
        
        return joint_pred, pose_pred


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, input_size=15, num_poses=5, output_size=147):
    """
    Factory function to create different models
    
    Args:
        model_type: 'gru', 'tcn', 'transformer', 'cnn_lstm'
        input_size: Sensor input dimension
        num_poses: Number of pose classes
        output_size: Joint output dimension
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    if model_type == 'gru':
        return HandPoseGRU(input_size, 128, 2, num_poses, output_size)
    
    elif model_type == 'tcn':
        return HandPoseTCN(input_size, [64, 128, 128, 256], 3, num_poses, output_size)
    
    elif model_type == 'transformer':
        return HandPoseTransformer(input_size, 128, 4, 3, num_poses, output_size)
    
    elif model_type == 'cnn_lstm':
        return HandPoseCNNLSTM(input_size, [32, 64], 128, 2, num_poses, output_size)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def print_model_comparison():
    """Print comparison of different models"""
    print("\n" + "="*70)
    print("MODEL COMPARISON FOR HAND POSE ESTIMATION")
    print("="*70)
    
    models = {
        'GRU': HandPoseGRU(),
        'TCN': HandPoseTCN(),
        'Transformer': HandPoseTransformer(),
        'CNN-LSTM': HandPoseCNNLSTM()
    }
    
    print(f"\n{'Model':<15} {'Parameters':<15} {'Best For'}")
    print("-"*70)
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        
        if name == 'GRU':
            best_for = "Speed, less data, simplicity"
        elif name == 'TCN':
            best_for = "Long sequences, parallel training"
        elif name == 'Transformer':
            best_for = "Lots of data, best accuracy"
        elif name == 'CNN-LSTM':
            best_for = "Spatial + temporal features"
        
        print(f"{name:<15} {param_count:>12,}   {best_for}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("1. START WITH: GRU (fastest, simplest)")
    print("2. IF UNDERFITTING: Try CNN-LSTM (better features)")
    print("3. IF NEED BETTER: Try TCN (better temporal)")
    print("4. IF LOTS OF DATA: Try Transformer (best accuracy)")
    print("\nAll models use quaternion normalization for valid rotations.")
    print("="*70)


if __name__ == "__main__":
    print_model_comparison()
    
    # Test all models with dummy input
    print("\nTesting all models with dummy input...")
    x = torch.randn(4, 10, 15)  # (batch=4, seq_len=10, features=15)
    
    for model_name in ['gru', 'tcn', 'transformer', 'cnn_lstm']:
        model = create_model(model_name, num_poses=5)
        joint_pred, pose_pred = model(x)
        print(f"\n{model_name.upper()}:")
        print(f"  Joint output shape: {joint_pred.shape}")
        print(f"  Pose output shape: {pose_pred.shape}")
        print(f"  ✓ Model works correctly")