"""
Flask API for Credit Risk Modeling with Graph Machine Learning
Based on Das et al. (2023) - Credit Risk Modeling with Graph Machine Learning
"""

import os
import hashlib
import json
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from database import CreditRiskDatabase
from data_loader import CreditRatingDataLoader

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize database and data loader
try:
    db = CreditRiskDatabase()
    data_loader = CreditRatingDataLoader("corporate_rating.csv")
    print("✅ Database and data loader initialized successfully!")
except Exception as e:
    print(f"⚠️ Warning during initialization: {e}")
    db = None
    data_loader = None

# Simple in-memory user storage (for demo)
users = {}

# Global model state
model_loaded = False
gnn_model = None
tabular_model = None


@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'name': 'Credit Risk Modeling with Graph Machine Learning',
        'version': '1.0.0',
        'based_on': 'Das et al. (2023) - INFORMS Journal on Data Science',
        'description': 'GNN-based credit risk assessment using corporate networks',
        'dataset': 'Kaggle Corporate Credit Rating (2,029 companies, 25 financial ratios)',
        'models': ['GraphSAGE GNN', 'AutoGluon Tabular', 'Ensemble'],
        'features': {
            'liquidity_ratios': 4,
            'profitability_ratios': 9,
            'debt_ratios': 2,
            'operating_performance_ratios': 5,
            'cash_flow_ratios': 5,
            'graph_features': 3
        }
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    stats = data_loader.get_statistics() if data_loader else {}
    db_stats = db.get_statistics() if db else {}
    
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db else 'not connected',
        'dataset_loaded': data_loader is not None and data_loader.df is not None,
        'total_companies': stats.get('total_companies', 0),
        'total_features': stats.get('total_features', 0),
        'total_predictions': db_stats.get('total_predictions', 0),
        'models': {
            'gnn_loaded': gnn_model is not None,
            'tabular_loaded': tabular_model is not None
        }
    })


@app.route('/api/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json
    
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')
    
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    
    if email in users:
        return jsonify({'error': 'User already exists'}), 409
    
    # Hash password
    hashed = hashlib.sha256(password.encode()).hexdigest()
    
    users[email] = {
        'name': name,
        'email': email,
        'password': hashed
    }
    
    return jsonify({
        'success': True,
        'message': 'Registration successful',
        'user': {'name': name, 'email': email}
    })


@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    
    email = data.get('email')
    password = data.get('password')
    
    if email not in users:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    hashed = hashlib.sha256(password.encode()).hexdigest()
    
    if users[email]['password'] == hashed:
        return jsonify({
            'success': True,
            'user': {
                'name': users[email]['name'],
                'email': email
            }
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401


@app.route('/api/companies', methods=['GET'])
def get_companies():
    """Get all companies from dataset"""
    limit = int(request.args.get('limit', 100))
    
    if data_loader is None or data_loader.df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    companies = []
    for idx, row in data_loader.df.head(limit).iterrows():
        company_data = {
            'id': int(idx),
            'name': row.get('Name', f'Company_{idx}'),
            'ticker': row.get('Symbol', 'N/A'),
            'rating': row.get('Rating', 'N/A'),
            'industry': row.get('Sector', 'N/A')
        }
        
        # Add key financial metrics
        if 'returnOnAssets' in row:
            try:
                company_data['roa'] = float(row['returnOnAssets'])
            except Exception:
                pass
        if 'debtEquityRatio' in row:
            try:
                company_data['debt_equity'] = float(row['debtEquityRatio'])
            except Exception:
                pass
        if 'currentRatio' in row:
            try:
                company_data['current_ratio'] = float(row['currentRatio'])
            except Exception:
                pass
        
        companies.append(company_data)
    
    return jsonify({
        'companies': companies,
        'total': len(companies)
    })


@app.route('/api/company/<int:company_id>', methods=['GET'])
def get_company_details(company_id):
    """Get detailed information about a specific company"""
    if data_loader is None:
        return jsonify({'error': 'Dataset not loaded'}), 500

    company_data = data_loader.get_company_data(company_id=company_id)
    
    if not company_data:
        return jsonify({'error': 'Company not found'}), 404
    
    # Clean the data for JSON serialization
    cleaned_data = {}
    for key, value in company_data.items():
        try:
            if pd.notna(value):
                if isinstance(value, (np.integer, np.floating)):
                    cleaned_data[key] = float(value)
                else:
                    cleaned_data[key] = str(value)
        except Exception:
            cleaned_data[key] = str(value)
    
    return jsonify({
        'success': True,
        'company': cleaned_data
    })


@app.route('/api/predict', methods=['POST'])
def predict_credit_risk():
    """
    Predict credit risk for a company
    Accepts either company_id or financial ratios
    """
    # Check if data_loader is available
    if data_loader is None:
        return jsonify({
            'error': 'System not initialized',
            'message': 'Dataset not loaded. Please check if corporate_rating.csv exists.'
        }), 503
    
    try:
        data = request.json
        
        # Extract company information
        company_id = data.get('company_id')
        company_name = data.get('company_name', 'Unknown Company')
        
        # Get or prepare features
        if company_id is not None:
            # Get from dataset
            company_data = data_loader.get_company_data(company_id=company_id)
            if not company_data:
                return jsonify({'error': 'Company not found'}), 404
            
            features = {k: v for k, v in company_data.items() 
                       if k in data_loader.feature_columns}
            company_name = company_data.get('Name', company_name)
        else:
            # Use provided features
            features = {}
            for k, v in data.items():
                if k in getattr(data_loader, 'feature_columns', []):
                    try:
                        features[k] = float(v)
                    except Exception:
                        pass
        
        if not features:
            return jsonify({'error': 'No valid features provided'}), 400
        
        # Simple credit score calculation
        credit_score = data_loader.predict_credit_score_simple(features)
        risk_level = data_loader.get_risk_level(credit_score)
        predicted_rating = data_loader.get_predicted_rating(credit_score)
        
        # Determine investment grade
        investment_grade = credit_score >= 700
        investment_grade_prob = min(max((credit_score - 300) / 600, 0), 1)
        
        # Generate prediction ID
        prediction_id = f'PRED{int(datetime.now().timestamp())}'
        
        # Save prediction to database
        try:
            prediction_data = {
                'company_id': str(company_id) if company_id else 'MANUAL',
                'prediction_id': prediction_id,
                'model_type': 'Simple Tabular',
                'model_version': '1.0',
                'predicted_rating': predicted_rating,
                'investment_grade_prob': investment_grade_prob,
                'below_investment_grade_prob': 1 - investment_grade_prob,
                'credit_score': credit_score,
                'risk_level': risk_level,
                'confidence_score': 0.75,
                'features': features,
                'used_graph': False,
                'graph_cutoff': None
            }
            if db:
                db.save_prediction(prediction_data)
        except Exception as e:
            print(f"Error saving prediction: {e}")
        
        # Generate recommendations
        recommendations = []
        if 'debtRatio' in features and features.get('debtRatio', 0) > 0.6:
            recommendations.append('Consider reducing debt ratio below 60%')
        if 'returnOnAssets' in features and features.get('returnOnAssets', 0) < 0.05:
            recommendations.append('Work on improving return on assets')
        if 'currentRatio' in features and features.get('currentRatio', 0) < 1.5:
            recommendations.append('Increase liquidity - target current ratio above 1.5')
        if 'debtEquityRatio' in features and features.get('debtEquityRatio', 0) > 2.0:
            recommendations.append('High leverage - consider reducing debt-to-equity ratio')
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'company_name': company_name,
            'credit_score': round(credit_score, 2),
            'predicted_rating': predicted_rating,
            'risk_level': risk_level,
            'investment_grade': investment_grade,
            'investment_grade_probability': round(investment_grade_prob, 4),
            'model_used': 'Simple Tabular Model',
            'model_type': 'baseline',
            'features_used': len(features),
            'recommendations': recommendations,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'explanation': {
                'method': 'Weighted financial ratio scoring',
                'key_factors': [
                    'Return on Assets',
                    'Debt Ratio',
                    'Current Ratio',
                    'Operating Cash Flow'
                ]
            }
        })
    
    except Exception as e:
        # Catch any errors in the prediction process
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error making prediction'
        }), 500


@app.route('/api/predict/gnn', methods=['POST'])
def predict_gnn():
    """
    Predict using Graph Neural Network
    (Placeholder - would require trained GNN model)
    """
    data = request.json
    company_id = data.get('company_id')
    
    if gnn_model is None:
        return jsonify({
            'error': 'GNN model not loaded',
            'message': 'Please train the GNN model first using /api/train/gnn'
        }), 503
    
    # This would use the actual GNN model
    return jsonify({
        'error': 'GNN prediction not yet implemented',
        'message': 'Use /api/predict for baseline predictions'
    }), 501


@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get prediction history"""
    limit = int(request.args.get('limit', 50))
    company_id = request.args.get('company_id')
    
    predictions = db.get_predictions(company_id=company_id, limit=limit) if db else []
    
    return jsonify({
        'predictions': predictions,
        'count': len(predictions)
    })


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall statistics"""
    dataset_stats = data_loader.get_statistics() if data_loader else {}
    db_stats = db.get_statistics() if db else {}
    
    return jsonify({
        'dataset': {
            'total_companies': dataset_stats.get('total_companies', 0),
            'total_features': dataset_stats.get('total_features', 0),
            'investment_grade_ratio': round(dataset_stats.get('investment_grade_ratio', 0) * 100, 1),
            'rating_distribution': dataset_stats.get('rating_distribution', {})
        },
        'predictions': {
            'total_predictions': db_stats.get('total_predictions', 0),
            'average_credit_score': db_stats.get('average_credit_score', 0),
            'risk_distribution': db_stats.get('risk_distribution', {}),
            'rating_distribution': db_stats.get('rating_distribution', {})
        },
        'graph': {
            'total_edges': db_stats.get('graph_edges', 0),
            'average_similarity': db_stats.get('average_similarity', 0)
        }
    })


@app.route('/api/dataset/info', methods=['GET'])
def dataset_info():
    """Get detailed dataset information"""
    stats = data_loader.get_statistics() if data_loader else {}
    
    return jsonify({
        'source': 'Kaggle Corporate Credit Rating Dataset',
        'paper': 'Das et al. (2023) - Credit Risk Modeling with Graph Machine Learning',
        'url': 'https://www.kaggle.com/agewerc/corporate-credit-rating',
        'statistics': stats,
        'features': {
            'liquidity_ratios': getattr(data_loader, 'ratio_features', {}).get('liquidity', []),
            'profitability_ratios': getattr(data_loader, 'ratio_features', {}).get('profitability', []),
            'debt_ratios': getattr(data_loader, 'ratio_features', {}).get('debt', []),
            'operating_performance': getattr(data_loader, 'ratio_features', {}).get('operating_performance', []),
            'cash_flow_ratios': getattr(data_loader, 'ratio_features', {}).get('cash_flow', [])
        },
        'methodology': {
            'graph_construction': 'SEC Filing MD&A Similarity (doc2vec + cosine similarity)',
            'gnn_architecture': 'GraphSAGE with 2 GCN layers',
            'hidden_dimension': 32,
            'training_method': 'Both transductive and inductive',
            'baseline_models': 'AutoGluon ensemble'
        }
    })


# Serve static files (frontend)
@app.route('/app')
def serve_app():
    """Serve the frontend application"""
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    print("=" * 80)
    print("🎯 Credit Risk Modeling with Graph Machine Learning")
    print("=" * 80)
    
    # Print dataset summary if available
    if data_loader:
        dataset_stats = data_loader.get_statistics()
        if dataset_stats:
            print(f"\n📊 Dataset Information:")
            print(f"   Total Companies: {dataset_stats.get('total_companies', 0)}")
            print(f"   Total Features: {dataset_stats.get('total_features', 0)}")
            print(f"   Investment Grade Ratio: {dataset_stats.get('investment_grade_ratio', 0)*100:.1f}%")
            if 'rating_distribution' in dataset_stats:
                print(f"\n📈 Rating Distribution:")
                for rating, count in sorted(dataset_stats['rating_distribution'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]:
                    print(f"      {rating}: {count}")
    else:
        print(f"\n⚠️  Dataset not found. Please download from:")
        print(f"   https://www.kaggle.com/agewerc/corporate-credit-rating")
        print(f"   Place as: corporate_rating.csv")
    
    print("\n" + "=" * 80)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
