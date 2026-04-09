"""
GNN Model for Crystal Structure Classification (Topological Insulator vs Trivial)
Uses PyTorch Geometric for Graph Neural Networks
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
import pickle
from typing import List, Optional
import glob

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
CONFIG = {
    'hidden_dim': 128,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'num_epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# Elemental features - atomic properties
ELEMENTAL_PROPERTIES = [
    'AtomicNumber', 'AtomicMass', 'X', 'Electronegativity', 
    'AtomicRadius', 'ElectronAffinity', 'IonizationEnergy',
    'MeltingPoint', 'BoilingPoint', 'Density',
    'Block', 'Group', 'Period'
]

def get_elemental_features(element: str) -> np.ndarray:
    """Extract elemental features for a given element symbol."""
    try:
        el = Element(element)
        features = [
            el.Z if el.Z else 0,  # Atomic Number
            el.atomic_mass if el.atomic_mass else 0,  # Atomic Mass
            el.X if el.X else 0,  # Electronegativity
            el.atomic_radius if el.atomic_radius else 0,  # Atomic Radius
            el.electron_affinity if el.electron_affinity else 0,  # Electron Affinity
            el.ionization_energy if el.ionization_energy else 0,  # Ionization Energy
            el.melting_point if el.melting_point else 0,  # Melting Point
            el.boiling_point if el.boiling_point else 0,  # Boiling Point
            el.density if el.density else 0,  # Density
            {'s': 0, 'p': 1, 'd': 2, 'f': 3}.get(el.block, 0),  # Block
            el.group if el.group else 0,  # Group
            el.period if el.period else 0  # Period
        ]
        return np.array(features, dtype=np.float32)
    except:
        return np.zeros(11, dtype=np.float32)

def parse_cif_to_graph(cif_path: str, cutoff: float = 5.0) -> Optional[Data]:
    """Parse CIF file and convert to graph structure."""
    try:
        structure = Structure.from_file(cif_path)
        
        # Get node features (atomic properties)
        elements = [str(site.specie) for site in structure]
        node_features = np.array([get_elemental_features(el) for el in elements])
        
        # Normalize features
        node_features = (node_features - node_features.mean(axis=0)) / (node_features.std(axis=0) + 1e-8)
        
        # Get node positions
        positions = np.array([site.coords for site in structure])
        num_nodes = len(elements)
        
        # Build edges based on distance (using periodic boundary conditions)
        edge_index = []
        edge_attr = []
        
        # Simple distance-based edge creation (without periodic boundaries for simplicity)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    # Edge features: distance
                    edge_attr.append([dist])
                    edge_attr.append([dist])
        
        if len(edge_index) == 0:
            # If no edges, create self-loops
            edge_index = [[i, i] for i in range(num_nodes)]
            edge_attr = [[0.0] for _ in range(num_nodes)]
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(positions, dtype=torch.float32)
        )
    except Exception as e:
        print(f"Error parsing {cif_path}: {e}")
        return None

class CrystalDataset(Dataset):
    """Custom dataset for crystal structures."""
    
    def __init__(self, csv_path: str, cif_folder: str, transform=None):
        super().__init__(None, transform)
        self.df = pd.read_csv(csv_path)
        self.cif_folder = cif_folder
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.df['label'].values)
        
        # Cache for graph data
        self.graph_cache = {}
        
    def len(self):
        return len(self.df)
    
    def get(self, idx):
        if idx in self.graph_cache:
            data = self.graph_cache[idx].clone()
        else:
            cif_path = os.path.join(self.cif_folder, self.df.iloc[idx]['cif_path'])
            data = parse_cif_to_graph(cif_path)
            
            if data is None:
                # Return empty graph if parsing fails
                data = Data(
                    x=torch.zeros((1, 11), dtype=torch.float32),
                    edge_index=torch.tensor([[0], [0]], dtype=torch.long),
                    edge_attr=torch.zeros((1, 1), dtype=torch.float32)
                )
            
            self.graph_cache[idx] = data.clone()
        
        data.y = torch.tensor(self.labels[idx], dtype=torch.long)
        data.num_atoms = len(data.x)
        return data

class GCNModel(nn.Module):
    """Graph Convolutional Network for crystal classification."""
    
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, num_layers: int = 4):
        super(GCNModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(in_channels, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Global pooling (mean and max)
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        # Concatenate pooling results
        x = torch.cat([mean_pool, max_pool], dim=1)
        
        # Output classification
        x = self.fc(x)
        return x

class GATModel(nn.Module):
    """Graph Attention Network for crystal classification."""
    
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int, 
                 num_layers: int = 3, num_heads: int = 4):
        super(GATModel, self).__init__()
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GATConv(in_channels, hidden_dim, heads=num_heads, dropout=0.3))
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.3))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GAT layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Global pooling
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        
        x = torch.cat([mean_pool, max_pool], dim=1)
        x = self.fc(x)
        return x

class CrystalGNN(nn.Module):
    """Combined GNN model with multiple aggregation methods."""
    
    def __init__(self, in_channels: int, hidden_dim: int, num_classes: int):
        super(CrystalGNN, self).__init__()
        
        # Initial embedding
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        
        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Input projection
        x = self.input_proj(x)
        
        # First GCN block
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        
        # Second GCN block
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        
        # Third GCN block
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        
        # Multi-scale pooling
        mean_pool = global_mean_pool(x3, batch)
        max_pool = global_max_pool(x3, batch)
        sum_pool = global_mean_pool(x1 + x2 + x3, batch)  # Sum pooling
        
        # Concatenate all pooled features
        x = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
        
        # Classification
        out = self.classifier(x)
        return out

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    
    return total_loss / total, correct / total

def train_gnn_model(csv_path: str, cif_folder: str, model_type: str = 'crystal_gnn'):
    """Main training function."""
    print("=" * 60)
    print(f"Training {model_type.upper()} Model")
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = CrystalDataset(csv_path, cif_folder)
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.label_encoder.classes_}")
    
    # Calculate class distribution
    unique, counts = np.unique(dataset.labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {dataset.label_encoder.inverse_transform([label])[0]}: {count}")
    
    # Split dataset
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, 
                                           stratify=dataset.labels)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42,
                                          stratify=[dataset.labels[i] for i in train_idx])
    
    print(f"\nSplit: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    
    # Create data loaders
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]
    test_dataset = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Get input dimensions
    sample_data = dataset[0]
    in_channels = sample_data.x.shape[1]
    num_classes = len(dataset.label_encoder.classes_)
    
    print(f"\nInput features: {in_channels}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    if model_type == 'gcn':
        model = GCNModel(in_channels, CONFIG['hidden_dim'], num_classes, CONFIG['num_layers'])
    elif model_type == 'gat':
        model = GATModel(in_channels, CONFIG['hidden_dim'], num_classes)
    else:
        model = CrystalGNN(in_channels, CONFIG['hidden_dim'], num_classes)
    
    model = model.to(device)
    print(f"\nModel: {model_type}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20
    
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model
    model_path = f"gnn_model_{model_type}.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'label_encoder': dataset.label_encoder,
        'config': CONFIG
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, test_acc

def predict(model, cif_path: str, device: str = 'cpu'):
    """Predict the class for a single CIF file."""
    model.eval()
    data = parse_cif_to_graph(cif_path)
    
    if data is None:
        return None
    
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        prob = F.softmax(out, dim=1)
    
    return pred.item(), prob.max().item()

if __name__ == "__main__":
    # Example usage
    csv_path = "Dataset/final_dataset.csv"
    cif_folder = "Dataset"
    
    # Train CrystalGNN model
    model, test_acc = train_gnn_model(csv_path, cif_folder, model_type='crystal_gnn')
    
    print(f"\nTraining completed! Test accuracy: {test_acc:.4f}")
