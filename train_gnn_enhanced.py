"""
Enhanced Training Script for Graph Neural Network Credit Risk Model
Includes: 60/20/20 split, K-Fold Cross-Validation, and SHAP Analysis
Based on Das et al. (2023) methodology
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score
from typing import Dict, List, Tuple
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import SHAP for model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

from data_loader import CreditRatingDataLoader
from gnn_model import GraphSAGE, GNNTrainer, CorporateGraph
from database import db


class EnhancedGNNTrainer:
    """Enhanced trainer with cross-validation and SHAP analysis"""
    
    def __init__(self, hidden_dim: int = 32, num_classes: int = 2, random_seed: int = 42):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.cv_results = []
        
    def prepare_data_splits(self, X: pd.DataFrame, y: pd.Series, 
                           train_size: float = 0.6, 
                           val_size: float = 0.2,
                           test_size: float = 0.2) -> Dict:
        """
        Split data into train (60%), validation (20%), and test (20%) sets
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
        
        # First split: separate test set (20%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, stratify=y
        )
        
        # Second split: separate train (60%) and validation (20%) from remaining 80%
        # validation should be 20% of total, which is 0.2/0.8 = 0.25 of temp
        val_size_adjusted = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_seed, stratify=y_temp
        )
        
        print(f"\n📊 Data Split Summary:")
        print(f"   Training:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Testing:    {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"   Total:      {len(X):,} samples")
        
        # Check class distribution
        print(f"\n   Class Distribution:")
        print(f"   Train - Default: {(y_train==1).sum()}, Non-Default: {(y_train==0).sum()}")
        print(f"   Val   - Default: {(y_val==1).sum()}, Non-Default: {(y_val==0).sum()}")
        print(f"   Test  - Default: {(y_test==1).sum()}, Non-Default: {(y_test==0).sum()}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      n_folds: int = 5,
                      epochs: int = 100,
                      learning_rate: float = 0.02,
                      similarity_threshold: float = 0.5) -> Dict:
        """
        Perform K-Fold cross-validation
        """
        print(f"\n🔄 Starting {n_folds}-Fold Cross-Validation...")
        print("=" * 80)
        
        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
        
        cv_scores = {
            'f1': [], 'accuracy': [], 'roc_auc': [], 'mcc': [],
            'precision': [], 'recall': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\n📁 Fold {fold}/{n_folds}")
            print("-" * 40)
            
            # Split data
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Build graph
            all_features = pd.concat([X_train_fold, X_val_fold])
            company_ids = [f"COMP_{i}" for i in all_features.index]
            
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(all_features.values)
            
            corp_graph = CorporateGraph()
            corp_graph.build_graph_from_similarity(company_ids, similarity_matrix, similarity_threshold)
            
            # Add graph features
            graph_features = corp_graph.compute_graph_features()
            for feature_name, values in graph_features.items():
                train_values = values[:len(X_train_fold)]
                val_values = values[len(X_train_fold):]
                X_train_fold = X_train_fold.copy()
                X_val_fold = X_val_fold.copy()
                X_train_fold[feature_name] = train_values
                X_val_fold[feature_name] = val_values
            
            # Prepare for GNN
            all_features_array = pd.concat([X_train_fold, X_val_fold]).values
            all_labels = pd.concat([y_train_fold, y_val_fold]).values
            
            n_train = len(X_train_fold)
            n_total = len(all_features_array)
            
            train_mask = np.zeros(n_total, dtype=bool)
            train_mask[:n_train] = True
            val_mask = np.zeros(n_total, dtype=bool)
            val_mask[n_train:] = True
            
            data = corp_graph.to_pytorch_geometric(
                all_features_array, all_labels, train_mask, val_mask
            )
            
            # Train model
            model = GraphSAGE(
                num_features=data.num_node_features,
                hidden_dim=self.hidden_dim,
                num_classes=self.num_classes
            )
            
            trainer = GNNTrainer(model)
            
            train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
            val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
            
            history = trainer.train(
                data=data,
                train_mask=train_mask_tensor,
                val_mask=val_mask_tensor,
                epochs=epochs,
                learning_rate=learning_rate,
                verbose=False
            )
            
            # Evaluate
            metrics = trainer.evaluate(data, val_mask_tensor)
            
            # Store results
            cv_scores['f1'].append(metrics['f1_score'])
            cv_scores['accuracy'].append(metrics['accuracy'])
            cv_scores['roc_auc'].append(metrics['roc_auc'])
            cv_scores['mcc'].append(metrics['mcc'])
            cv_scores['precision'].append(metrics['precision'])
            cv_scores['recall'].append(metrics['recall'])
            
            print(f"   F1: {metrics['f1_score']:.4f}, Accuracy: {metrics['accuracy']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Calculate statistics
        print(f"\n{'='*80}")
        print(f"📊 Cross-Validation Results Summary")
        print(f"{'='*80}")
        
        cv_summary = {}
        for metric_name, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            cv_summary[metric_name] = {
                'mean': mean_score,
                'std': std_score,
                'scores': scores
            }
            print(f"   {metric_name.upper():12s}: {mean_score:.4f} ± {std_score:.4f}")
        
        print(f"{'='*80}")
        
        self.cv_results = cv_summary
        return cv_summary
    
    def train_final_model(self, splits: Dict, 
                         similarity_threshold: float = 0.5,
                         epochs: int = 120,
                         learning_rate: float = 0.02,
                         verbose: bool = True) -> Tuple:
        """
        Train final model on train set, validate on val set, test on test set
        """
        print(f"\n🧠 Training Final Model...")
        print("=" * 80)
        
        X_train = splits['X_train']
        X_val = splits['X_val']
        X_test = splits['X_test']
        y_train = splits['y_train']
        y_val = splits['y_val']
        y_test = splits['y_test']
        
        # Build graph with all data
        all_features = pd.concat([X_train, X_val, X_test])
        company_ids = [f"COMP_{i}" for i in all_features.index]
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(all_features.values)
        
        corp_graph = CorporateGraph()
        corp_graph.build_graph_from_similarity(company_ids, similarity_matrix, similarity_threshold)
        
        print(f"   Graph: {corp_graph.num_nodes()} nodes, {corp_graph.num_edges()} edges")
        
        # Add graph features
        graph_features = corp_graph.compute_graph_features()
        n_train = len(X_train)
        n_val = len(X_val)
        
        for feature_name, values in graph_features.items():
            X_train = X_train.copy()
            X_val = X_val.copy()
            X_test = X_test.copy()
            X_train[feature_name] = values[:n_train]
            X_val[feature_name] = values[n_train:n_train+n_val]
            X_test[feature_name] = values[n_train+n_val:]
        
        print(f"   Features: {len(X_train.columns)} (including 3 graph features)")
        
        # Prepare for GNN
        all_features_array = pd.concat([X_train, X_val, X_test]).values
        all_labels = pd.concat([y_train, y_val, y_test]).values
        
        n_total = len(all_features_array)
        
        train_mask = np.zeros(n_total, dtype=bool)
        train_mask[:n_train] = True
        
        val_mask = np.zeros(n_total, dtype=bool)
        val_mask[n_train:n_train+n_val] = True
        
        test_mask = np.zeros(n_total, dtype=bool)
        test_mask[n_train+n_val:] = True
        
        data = corp_graph.to_pytorch_geometric(
            all_features_array, all_labels, train_mask, test_mask
        )
        
        # Initialize model
        model = GraphSAGE(
            num_features=data.num_node_features,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes
        )
        
        trainer = GNNTrainer(model)
        
        # Train
        train_mask_tensor = torch.tensor(train_mask, dtype=torch.bool)
        val_mask_tensor = torch.tensor(val_mask, dtype=torch.bool)
        test_mask_tensor = torch.tensor(test_mask, dtype=torch.bool)
        
        print(f"\n   Training for {epochs} epochs...")
        history = trainer.train(
            data=data,
            train_mask=train_mask_tensor,
            val_mask=val_mask_tensor,
            epochs=epochs,
            learning_rate=learning_rate,
            verbose=verbose
        )
        
        # Evaluate on all sets
        train_metrics = trainer.evaluate(data, train_mask_tensor)
        val_metrics = trainer.evaluate(data, val_mask_tensor)
        test_metrics = trainer.evaluate(data, test_mask_tensor)
        
        print(f"\n{'='*80}")
        print(f"📊 Final Model Performance")
        print(f"{'='*80}")
        print(f"{'Metric':<15} {'Train':<12} {'Validation':<12} {'Test':<12}")
        print(f"{'-'*80}")
        print(f"{'F1 Score':<15} {train_metrics['f1_score']:<12.4f} {val_metrics['f1_score']:<12.4f} {test_metrics['f1_score']:<12.4f}")
        print(f"{'Accuracy':<15} {train_metrics['accuracy']:<12.4f} {val_metrics['accuracy']:<12.4f} {test_metrics['accuracy']:<12.4f}")
        print(f"{'ROC-AUC':<15} {train_metrics['roc_auc']:<12.4f} {val_metrics['roc_auc']:<12.4f} {test_metrics['roc_auc']:<12.4f}")
        print(f"{'MCC':<15} {train_metrics['mcc']:<12.4f} {val_metrics['mcc']:<12.4f} {test_metrics['mcc']:<12.4f}")
        print(f"{'Precision':<15} {train_metrics['precision']:<12.4f} {val_metrics['precision']:<12.4f} {test_metrics['precision']:<12.4f}")
        print(f"{'Recall':<15} {train_metrics['recall']:<12.4f} {val_metrics['recall']:<12.4f} {test_metrics['recall']:<12.4f}")
        print(f"{'='*80}")
        
        return model, trainer, data, (train_metrics, val_metrics, test_metrics), history, corp_graph, X_train.columns.tolist()
    
    def compute_shap_values(self, model, data, feature_names: List[str],
                           train_mask, test_mask, 
                           n_background: int = 100,
                           n_explain: int = 100) -> Dict:
        """
        Compute SHAP values for model interpretability
        """
        if not SHAP_AVAILABLE:
            print("\n⚠️  SHAP not available. Skipping SHAP analysis.")
            return None
        
        print(f"\n🔍 Computing SHAP Values for Model Interpretability...")
        print("=" * 80)
        
        model.eval()
        
        # Get node embeddings and features
        with torch.no_grad():
            # Get predictions
            out = model(data.x, data.edge_index)
            probs = torch.softmax(out, dim=1)
        
        # Extract features for SHAP analysis
        train_indices = torch.where(train_mask)[0].numpy()
        test_indices = torch.where(test_mask)[0].numpy()
        
        # Sample background data
        if len(train_indices) > n_background:
            background_idx = np.random.choice(train_indices, n_background, replace=False)
        else:
            background_idx = train_indices
        
        # Sample test data to explain
        if len(test_indices) > n_explain:
            explain_idx = np.random.choice(test_indices, n_explain, replace=False)
        else:
            explain_idx = test_indices
        
        background_data = data.x[background_idx].numpy()
        explain_data = data.x[explain_idx].numpy()
        
        print(f"   Background samples: {len(background_idx)}")
        print(f"   Samples to explain: {len(explain_idx)}")
        
        # Create a prediction function for SHAP
        def predict_fn(X):
            """Wrapper function for SHAP"""
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                # For simplicity, we'll use the model's forward pass
                # In a full GNN, this would need the graph structure
                # Here we approximate with just the node features
                out = model.conv1(X_tensor, data.edge_index)
                out = torch.relu(out)
                out = model.conv2(out, data.edge_index)
                probs = torch.softmax(out, dim=1)
                return probs[:, 1].numpy()  # Probability of default
        
        try:
            # Use KernelExplainer (model-agnostic)
            print("   Initializing SHAP KernelExplainer...")
            explainer = shap.KernelExplainer(predict_fn, background_data)
            
            print("   Computing SHAP values (this may take a few minutes)...")
            shap_values = explainer.shap_values(explain_data, nsamples=100)
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                'Feature': feature_names[:len(feature_importance)],
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            print(f"\n   Top 10 Most Important Features:")
            print(f"   {'-'*50}")
            for idx, row in importance_df.head(10).iterrows():
                print(f"   {row['Feature']:<30s}: {row['Importance']:.4f}")
            
            shap_results = {
                'shap_values': shap_values,
                'background_data': background_data,
                'explain_data': explain_data,
                'feature_importance': importance_df,
                'explainer': explainer
            }
            
            print(f"\n   ✅ SHAP analysis complete!")
            
            return shap_results
            
        except Exception as e:
            print(f"\n   ⚠️  Error computing SHAP values: {e}")
            return None
    
    def plot_results(self, history: Dict, cv_results: Dict, 
                    shap_results: Dict = None, save_path: str = None):
        """
        Create visualization plots
        """
        print(f"\n📈 Creating Visualization Plots...")
        
        # Determine number of subplots
        n_plots = 3 if shap_results is not None else 2
        fig = plt.figure(figsize=(15, 5*((n_plots+1)//2)))
        
        # Plot 1: Training History
        ax1 = plt.subplot(2, 2, 1)
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Metrics
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
        ax2.plot(epochs, history['val_f1'], 'r-', label='Val F1', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cross-Validation Results
        ax3 = plt.subplot(2, 2, 3)
        metrics = ['f1', 'accuracy', 'roc_auc', 'mcc', 'precision', 'recall']
        means = [cv_results[m]['mean'] for m in metrics]
        stds = [cv_results[m]['std'] for m in metrics]
        
        x_pos = np.arange(len(metrics))
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
        ax3.set_xlabel('Metrics', fontsize=12)
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Cross-Validation Results (Mean ± Std)', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([m.upper() for m in metrics], rotation=45, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1])
        
        # Plot 4: Feature Importance (SHAP)
        if shap_results is not None:
            ax4 = plt.subplot(2, 2, 4)
            top_features = shap_results['feature_importance'].head(10)
            ax4.barh(range(len(top_features)), top_features['Importance'].values, color='coral')
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['Feature'].values)
            ax4.set_xlabel('SHAP Importance', fontsize=12)
            ax4.set_title('Top 10 Features (SHAP)', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')
            ax4.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Plot saved: {save_path}")
        
        return fig


def train_gnn_with_cv_and_shap(data_path: str = 'corporate_rating.csv',
                               similarity_threshold: float = 0.5,
                               hidden_dim: int = 32,
                               epochs: int = 120,
                               learning_rate: float = 0.02,
                               n_folds: int = 5,
                               random_seed: int = 42,
                               save_results: bool = True) -> Dict:
    """
    Complete training pipeline with cross-validation and SHAP
    """
    print("=" * 80)
    print("🚀 Enhanced GNN Credit Risk Model Training")
    print("   Features: 60/20/20 Split, Cross-Validation, SHAP Analysis")
    print("   Based on: Das et al. (2023)")
    print("=" * 80)
    
    # 1. Load Data
    print("\n📊 Step 1: Loading Dataset...")
    data_loader = CreditRatingDataLoader(data_path)
    
    if data_loader.df is None:
        raise ValueError(f"Could not load dataset from {data_path}")
    
    print(f"   Dataset: {len(data_loader.df)} companies")
    print(f"   Features: {len(data_loader.df.columns)} columns")
    
    # 2. Prepare Features
    print("\n🔧 Step 2: Preparing Features...")
    X, y = data_loader.prepare_features_labels()
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Default rate: {(y==1).sum()/len(y)*100:.2f}%")
    
    # 3. Initialize Enhanced Trainer
    trainer = EnhancedGNNTrainer(
        hidden_dim=hidden_dim,
        num_classes=2,
        random_seed=random_seed
    )
    
    # 4. Create Data Splits (60/20/20)
    print("\n✂️  Step 3: Creating Data Splits (60% Train, 20% Val, 20% Test)...")
    splits = trainer.prepare_data_splits(X, y, train_size=0.6, val_size=0.2, test_size=0.2)
    
    # 5. Cross-Validation
    print("\n🔄 Step 4: Performing Cross-Validation...")
    cv_results = trainer.cross_validate(
        X=splits['X_train'],
        y=splits['y_train'],
        n_folds=n_folds,
        epochs=100,  # Use fewer epochs for CV
        learning_rate=learning_rate,
        similarity_threshold=similarity_threshold
    )
    
    # 6. Train Final Model
    print("\n🎯 Step 5: Training Final Model...")
    model, gnn_trainer, data, metrics_tuple, history, graph, feature_names = trainer.train_final_model(
        splits=splits,
        similarity_threshold=similarity_threshold,
        epochs=epochs,
        learning_rate=learning_rate,
        verbose=True
    )
    
    train_metrics, val_metrics, test_metrics = metrics_tuple
    
    # 7. SHAP Analysis
    print("\n🔍 Step 6: SHAP Analysis...")
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[:len(splits['X_train'])] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[len(splits['X_train'])+len(splits['X_val']):] = True
    
    shap_results = trainer.compute_shap_values(
        model=model,
        data=data,
        feature_names=feature_names,
        train_mask=train_mask,
        test_mask=test_mask,
        n_background=100,
        n_explain=100
    )
    
    # 8. Create Visualizations
    print("\n📊 Step 7: Creating Visualizations...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_path = f'/mnt/user-data/outputs/training_results_{timestamp}.png'
    
    fig = trainer.plot_results(history, cv_results, shap_results, save_path=plot_path)
    
    # 9. Save Results
    if save_results:
        print("\n💾 Step 8: Saving Results...")
        
        # Save model
        model_path = f'/mnt/user-data/outputs/gnn_model_{timestamp}.pt'
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
                'n_folds': n_folds
            },
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'cv_results': cv_results,
            'feature_names': feature_names
        }, model_path)
        print(f"   Model saved: {model_path}")
        
        # Save results summary
        results_path = f'/mnt/user-data/outputs/results_summary_{timestamp}.json'
        results_summary = {
            'timestamp': timestamp,
            'data_splits': {
                'train_size': len(splits['X_train']),
                'val_size': len(splits['X_val']),
                'test_size': len(splits['X_test'])
            },
            'cross_validation': {
                metric: {
                    'mean': float(cv_results[metric]['mean']),
                    'std': float(cv_results[metric]['std'])
                }
                for metric in cv_results.keys()
            },
            'final_model': {
                'train': {k: float(v) for k, v in train_metrics.items()},
                'validation': {k: float(v) for k, v in val_metrics.items()},
                'test': {k: float(v) for k, v in test_metrics.items()}
            },
            'hyperparameters': {
                'hidden_dim': hidden_dim,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'similarity_threshold': similarity_threshold,
                'n_folds': n_folds
            }
        }
        
        if shap_results is not None:
            results_summary['feature_importance'] = shap_results['feature_importance'].head(20).to_dict('records')
        
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"   Results summary saved: {results_path}")
        
        # Save to database
        db.save_model_performance({
            'model_name': 'GraphSAGE_GNN_Enhanced',
            'model_type': 'GNN_with_CV_and_SHAP',
            'f1_score': test_metrics['f1_score'],
            'accuracy': test_metrics['accuracy'],
            'roc_auc': test_metrics['roc_auc'],
            'mcc': test_metrics['mcc'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'train_size': len(splits['X_train']),
            'test_size': len(splits['X_test']),
            'dataset_name': 'Kaggle Corporate Credit Rating',
            'hyperparameters': {
                'hidden_dim': hidden_dim,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'similarity_threshold': similarity_threshold,
                'n_folds': n_folds,
                'cv_mean_f1': float(cv_results['f1']['mean']),
                'cv_std_f1': float(cv_results['f1']['std'])
            }
        })
        print(f"   Results saved to database")
    
    print(f"\n{'='*80}")
    print(f"✅ Training Pipeline Complete!")
    print(f"{'='*80}")
    
    return {
        'model': model,
        'trainer': gnn_trainer,
        'data': data,
        'splits': splits,
        'metrics': metrics_tuple,
        'cv_results': cv_results,
        'shap_results': shap_results,
        'history': history,
        'graph': graph,
        'feature_names': feature_names,
        'model_path': model_path if save_results else None,
        'plot_path': plot_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced GNN Training with Cross-Validation and SHAP'
    )
    parser.add_argument('--data_path', type=str, default='/mnt/project/corporate_rating.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Similarity threshold for graph edges (0.5-0.7)')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension for GNN')
    parser.add_argument('--epochs', type=int, default=120,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.02,
                       help='Learning rate')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results')
    
    args = parser.parse_args()
    
    results = train_gnn_with_cv_and_shap(
        data_path=args.data_path,
        similarity_threshold=args.threshold,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        n_folds=args.n_folds,
        random_seed=args.seed,
        save_results=not args.no_save
    )
    
    return results


if __name__ == '__main__':
    main()