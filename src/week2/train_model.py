"""
NEURA Glove - Model Training
Trains LSTM network on collected sensor→landmark data
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from glob import glob


class HandPoseDataset(Dataset):
    """Dataset of sensor sequences → hand landmarks"""
    
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        
        # Load all training files
        files = sorted(glob(f"{data_dir}/*.json"))
        print(f"Loading {len(files)} samples...")
        
        for filepath in files:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Only use high-confidence samples
            if data['ground_truth']['confidence'] > 0.7:
                self.samples.append(data)
        
        print(f"Loaded {len(self.samples)} valid samples")
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.samples) - sequence_length):
            seq_data = []
            for j in range(sequence_length):
                s = self.samples[i + j]
                # Combine all sensor features: 5 flex + 3 accel + 3 gyro + 4 quat = 15
                features = np.concatenate([
                    s['sensor']['flex'],
                    s['sensor']['accel'],
                    s['sensor']['gyro'],
                    s['sensor']['quat']
                ])
                seq_data.append(features)
            
            # Target: landmarks at end of sequence (21×3 = 63 values)
            target = np.array(self.samples[i + sequence_length]['ground_truth']['landmarks']).flatten()
            
            self.sequences.append((np.array(seq_data), target))
        
        print(f"Created {len(self.sequences)} training sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.FloatTensor(seq), torch.FloatTensor(target)


class LSTMPoseModel(nn.Module):
    """LSTM network: sensor sequences → joint positions"""
    
    def __init__(self, input_size=15, hidden_size=128, num_layers=2, output_size=63):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 15)
        lstm_out, _ = self.lstm(x)
        final_hidden = lstm_out[:, -1, :]  # Last timestep
        output = self.decoder(final_hidden)
        return output


def train_model(data_dir, epochs=50, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the LSTM model"""
    
    print("="*60)
    print("NEURA GLOVE MODEL TRAINING")
    print("="*60)
    print(f"Device: {device}")
    print(f"Data directory: {data_dir}\n")
    
    # Load dataset
    dataset = HandPoseDataset(data_dir)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}\n")
    
    # Create model
    model = LSTMPoseModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    
    print("Starting training...\n")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'trained_model.pth')
            print(f"  → Model saved (val_loss: {val_loss:.6f})")
    
    print(f"\n✓ Training complete!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: trained_model.pth")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    train_model(args.data, epochs=args.epochs, batch_size=args.batch_size)