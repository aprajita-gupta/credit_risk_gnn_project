"""
Training Script for Graph Neural Network Credit Risk Model
Based on Das et al. (2023) methodology
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, List
import argparse
import json
from datetime import datetime

from data_loader import CreditRatingDataLoader
from gnn_model import GraphSAGE, GNNTrainer, CorporateGraph, create_similarity_matrix_from_text
from database import db


def train_gnn_model(data_path: str = 'corporate_credit_rating.csv',
                   similarity_threshold: float = 0.5,
                   hidden_dim: int = 32,
                   epochs: int = 120,
                   learning_rate: float = 0.02,
                   test_size: float = 0.2,
                   random_seed: int = 42,
                   use_graph_features: bool = True,
                   training_mode: str = 'transductive',
                   verbose: bool = True) -> Dict:
    """
    Train GNN model for credit risk classification
    
    Args:
        data_path: Path to dataset CSV
        similarity_threshold: Threshold for graph edge creation (0.5-0.7 per paper)
        hidden_dim: Hidden dimension for GNN (32 per paper)
        epochs: Number of training epochs (120 per paper)
        learning_rate: Learning rate (0.02 per paper)
        test_size: Test set ratio
        random_seed: Random seed for reproducibility
        use_graph_features: Whether to add graph features to tabular data
        training_mode: 'transductive' or 'inductive'
        verbose: Print training progress
    
    Returns:
        Dictionary with model, metrics, and training info
    """
    
    print("=" * 80)
    print("🚀 Training GNN Credit Risk Model")
    print("   Based on: Das et al. (2023) - Credit Risk Modeling with Graph ML")
    print("=" * 80)
    
    # 1. Load Data
    print("\n📊 Step 1: Loading dataset...")
    data_loader = CreditRatingDataLoader(data_path)
    
    if data_loader.df is None:
        raise ValueError(f"Could not load dataset from {data_path}")
    
    # 2. Prepare features and labels
    print("\n🔧 Step 2: Preparing features and labels...")
    split_data = data_loader.prepare_train_test_split(
        test_size=test_size,
        random_state=random_seed,
        add_graph_features=False  # Will add later from actual graph
    )
    
    X_train = split_data['X_train']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_test = split_data['y_test']
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(split_data['feature_names'])}")
    
    # 3. Construct Corporate Graph
    print(f"\n🌐 Step 3: Constructing corporate graph (threshold={similarity_threshold})...")
    
    # Get company IDs
    all_indices = list(X_train.index) + list(X_test.index)
    company_ids = [f"COMP_{i}" for i in all_indices]
    
    # Create similarity matrix (in production, would use SEC filings)
    # For now, use feature-based similarity as proxy
    all_features = pd.concat([X_train, X_test])
    from sklearn.metrics.pairwise import cosine_similarity
    similarity_matrix = cosine_similarity(all_features.values)
    
    # Build graph
    corp_graph = CorporateGraph()
    corp_graph.build_graph_from_similarity(company_ids, similarity_matrix, similarity_threshold)
    
    # 4. Compute graph features
    if use_graph_features:
        print("\n📈 Step 4: Computing graph features...")
        graph_features = corp_graph.compute_graph_features()
        
        # Add to feature matrices
        for feature_name, values in graph_features.items():
            train_values = values[:len(X_train)]
            test_values = values[len(X_train):]
            
            X_train[feature_name] = train_values
            X_test[feature_name] = test_values
        
        print(f"   Added 3 graph features (degree, eigenvector, clustering)")
        print(f"   Total features now: {len(X_train.columns)}")
    
    # 5. Prepare data for GNN
    print(f"\n🔮 Step 5: Preparing data for GNN ({training_mode} mode)...")
    
    # Combine features and labels
    all_features = pd.concat([X_train, X_test]).values
    all_labels = pd.concat([y_train, y_test]).values
    
    # Create masks
    n_total = len(all_features)
    n_train = len(X_train)
    
    if training_mode == 'transductive':
        # Use all nodes in graph, but only train labels
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train:] = True
    else:  # inductive
        # Only use training nodes in graph
        # For simplicity, we still use full graph but note this difference
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train:] = True
    
    # Create PyTorch Geometric data
    data = corp_graph.to_pytorch_geometric(all_features, all_labels, train_mask, test_mask)
    
    print(f"   Nodes: {data.num_nodes}")
    print(f"   Edges: {data.num_edges}")
    print(f"   Features per node: {data.num_node_features}")
    
    # 6. Initialize and train model
    print(f"\n🧠 Step 6: Training GraphSAGE model...")
    print(f"   Architecture: 2 GCN layers, hidden_dim={hidden_dim}")
    print(f"   Training: {epochs} epochs, lr={learning_rate}")
    
    model = GraphSAGE(
        num_features=data.num_node_features,
        hidden_dim=hidden_dim,
        num_classes=2
    )
    
    trainer = GNNTrainer(model)
    
    # Create validation split from training data
    train_indices = np.where(train_mask)[0]
    train_idx, val_idx = train_test_split(train_indices, test_size=0.2, random_state=random_seed)
    
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_idx] = True
    val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
    
    train_mask_final = np.zeros(n_total, dtype=bool)
    train_mask_final[train_idx] = True
    train_mask_tensor = torch.tensor(train_mask_final, dtype=torch.bool)
    
    # Train
    history = trainer.train(
        data=data,
        train_mask=train_mask_tensor,
        val_mask=val_mask_tensor,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=verbose
    )
    
    # 7. Evaluate on test set
    print(f"\n📊 Step 7: Evaluating on test set...")
    test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
    test_metrics = trainer.evaluate(data, test_mask_tensor)
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Set Performance:")
    print(f"{'='*50}")
    print(f"   F1 Score:    {test_metrics['f1_score']:.4f}")
    print(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"   ROC-AUC:     {test_metrics['roc_auc']:.4f}")
    print(f"   MCC:         {test_metrics['mcc']:.4f}")
    print(f"   Precision:   {test_metrics['precision']:.4f}")
    print(f"   Recall:      {test_metrics['recall']:.4f}")
    print(f"{'='*50}")
    
    # 8. Save model and results
    print(f"\n💾 Step 8: Saving model and results...")
    
    # Save model weights
    model_path = f"gnn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_features': data.num_node_features,
            'hidden_dim': hidden_dim,
            'num_classes': 2
        },
        'training_config': {
            'epochs': epochs,
            'learning_rate': learning_rate,
            'similarity_threshold': similarity_threshold,
            'training_mode': training_mode
        },
        'metrics': test_metrics,
        'feature_names': split_data['feature_names']
    }, model_path)
    
    print(f"   Model saved: {model_path}")
    
    # Save to database
    db.save_model_performance({
        'model_name': 'GraphSAGE_GNN',
        'model_type': f'GNN_{training_mode}',
        'f1_score': test_metrics['f1_score'],
        'accuracy': test_metrics['accuracy'],
        'roc_auc': test_metrics['roc_auc'],
        'mcc': test_metrics['mcc'],
        'precision': test_metrics['precision'],
        'recall': test_metrics['recall'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'dataset_name': 'Kaggle Corporate Credit Rating',
        'hyperparameters': {
            'hidden_dim': hidden_dim,
            'epochs': epochs,
            'learning_rate': learning_rate,
            'similarity_threshold': similarity_threshold,
            'use_graph_features': use_graph_features
        }
    })
    
    print(f"\n✅ Training complete!")
    
    return {
        'model': model,
        'trainer': trainer,
        'data': data,
        'metrics': test_metrics,
        'history': history,
        'model_path': model_path,
        'graph': corp_graph
    }


def main():
    parser = argparse.ArgumentParser(description='Train GNN Credit Risk Model')
    parser.add_argument('--data_path', type=str, default='corporate_credit_rating.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for graph edges (0.5-0.7)')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension for GNN')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.02,
                       help='Learning rate')
    parser.add_argument('--mode', type=str, default='transductive',
                       choices=['transductive', 'inductive'],
                       help='Training mode')
    parser.add_argument('--no_graph_features', action='store_true',
                       help='Disable graph features in tabular data')
    
    args = parser.parse_args()
    
    results = train_gnn_model(
        data_path=args.data_path,
        similarity_threshold=args.threshold,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        training_mode=args.mode,
        use_graph_features=not args.no_graph_features,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    main()