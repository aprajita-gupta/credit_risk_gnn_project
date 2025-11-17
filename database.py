"""
Database Module for Credit Risk Modeling System
Based on Das et al. (2023) - Credit Risk Modeling with Graph Machine Learning
"""

import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
import json


class CreditRiskDatabase:
    """Database handler for credit risk predictions and company data"""
    
    def __init__(self, db_name='credit_risk_gnn.db'):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        """Initialize all required database tables"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Companies table - stores company financial data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS companies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT UNIQUE NOT NULL,
                company_name TEXT,
                ticker TEXT,
                industry TEXT,
                
                -- Liquidity Ratios
                current_ratio REAL,
                quick_ratio REAL,
                cash_ratio REAL,
                days_sales_outstanding REAL,
                
                -- Profitability Ratios
                gross_profit_margin REAL,
                operating_profit_margin REAL,
                pretax_profit_margin REAL,
                net_profit_margin REAL,
                effective_tax_rate REAL,
                return_on_assets REAL,
                return_on_equity REAL,
                return_on_capital_employed REAL,
                ebit_to_revenue REAL,
                
                -- Debt Ratios
                debt_ratio REAL,
                debt_equity_ratio REAL,
                
                -- Operating Performance Ratios
                asset_turnover REAL,
                fixed_asset_turnover REAL,
                company_equity_multiplier REAL,
                enterprise_value_multiple REAL,
                payable_turnover REAL,
                
                -- Cash Flow Ratios
                operating_cashflow_per_share REAL,
                free_cashflow_per_share REAL,
                cash_per_share REAL,
                operating_cashflow_sales_ratio REAL,
                free_cashflow_operating_ratio REAL,
                
                -- Graph Metrics
                degree_centrality REAL,
                eigenvector_centrality REAL,
                clustering_coefficient REAL,
                
                -- Actual Rating (if available)
                actual_rating TEXT,
                rating_agency TEXT,
                rating_date TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT,
                prediction_id TEXT UNIQUE NOT NULL,
                
                -- Model Information
                model_type TEXT,
                model_version TEXT,
                
                -- Prediction Results
                predicted_rating TEXT,
                investment_grade_prob REAL,
                below_investment_grade_prob REAL,
                credit_score REAL,
                risk_level TEXT,
                
                -- Model Metrics (if available)
                confidence_score REAL,
                
                -- Features used
                features_json TEXT,
                
                -- Graph information
                used_graph BOOLEAN DEFAULT 0,
                graph_cutoff REAL,
                
                prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (company_id) REFERENCES companies (company_id)
            )
        ''')
        
        # Graph edges table - stores corporate network relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_company_id TEXT,
                target_company_id TEXT,
                similarity_score REAL,
                edge_type TEXT DEFAULT 'SEC_FILING_SIMILARITY',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (source_company_id) REFERENCES companies (company_id),
                FOREIGN KEY (target_company_id) REFERENCES companies (company_id),
                UNIQUE(source_company_id, target_company_id)
            )
        ''')
        
        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                model_type TEXT,
                
                -- Performance Metrics
                f1_score REAL,
                accuracy REAL,
                roc_auc REAL,
                mcc REAL,
                precision_score REAL,
                recall_score REAL,
                
                -- Dataset Information
                train_size INTEGER,
                test_size INTEGER,
                dataset_name TEXT,
                
                -- Hyperparameters
                hyperparameters_json TEXT,
                
                training_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_company(self, company_data: Dict[str, Any]) -> bool:
        """Add or update a company in the database"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            columns = ', '.join(company_data.keys())
            placeholders = ', '.join(['?' for _ in company_data])
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO companies ({columns})
                VALUES ({placeholders})
            ''', tuple(company_data.values()))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding company: {e}")
            return False
        finally:
            conn.close()
    
    def get_company(self, company_id: str) -> Optional[Dict]:
        """Retrieve company data"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM companies WHERE company_id = ?
        ''', (company_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return dict(result)
        return None
    
    def get_all_companies(self, limit: int = 100) -> List[Dict]:
        """Get all companies"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM companies LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def save_prediction(self, prediction_data: Dict[str, Any]) -> bool:
        """Save a credit risk prediction"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO predictions 
                (company_id, prediction_id, model_type, model_version,
                 predicted_rating, investment_grade_prob, below_investment_grade_prob,
                 credit_score, risk_level, confidence_score, features_json,
                 used_graph, graph_cutoff)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data.get('company_id'),
                prediction_data.get('prediction_id'),
                prediction_data.get('model_type'),
                prediction_data.get('model_version'),
                prediction_data.get('predicted_rating'),
                prediction_data.get('investment_grade_prob'),
                prediction_data.get('below_investment_grade_prob'),
                prediction_data.get('credit_score'),
                prediction_data.get('risk_level'),
                prediction_data.get('confidence_score'),
                json.dumps(prediction_data.get('features', {})),
                prediction_data.get('used_graph', False),
                prediction_data.get('graph_cutoff')
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
        finally:
            conn.close()
    
    def get_predictions(self, company_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get predictions, optionally filtered by company"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if company_id:
            cursor.execute('''
                SELECT * FROM predictions 
                WHERE company_id = ?
                ORDER BY prediction_timestamp DESC
                LIMIT ?
            ''', (company_id, limit))
        else:
            cursor.execute('''
                SELECT * FROM predictions 
                ORDER BY prediction_timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def add_graph_edge(self, source_id: str, target_id: str, similarity: float) -> bool:
        """Add an edge to the corporate graph"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO graph_edges 
                (source_company_id, target_company_id, similarity_score)
                VALUES (?, ?, ?)
            ''', (source_id, target_id, similarity))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding graph edge: {e}")
            return False
        finally:
            conn.close()
    
    def get_graph_edges(self, min_similarity: float = 0.0) -> List[Dict]:
        """Get all graph edges above a similarity threshold"""
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM graph_edges 
            WHERE similarity_score >= ?
        ''', (min_similarity,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def save_model_performance(self, performance_data: Dict[str, Any]) -> bool:
        """Save model performance metrics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO model_performance 
                (model_name, model_type, f1_score, accuracy, roc_auc, mcc,
                 precision_score, recall_score, train_size, test_size,
                 dataset_name, hyperparameters_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                performance_data.get('model_name'),
                performance_data.get('model_type'),
                performance_data.get('f1_score'),
                performance_data.get('accuracy'),
                performance_data.get('roc_auc'),
                performance_data.get('mcc'),
                performance_data.get('precision'),
                performance_data.get('recall'),
                performance_data.get('train_size'),
                performance_data.get('test_size'),
                performance_data.get('dataset_name'),
                json.dumps(performance_data.get('hyperparameters', {}))
            ))
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Error saving model performance: {e}")
            return False
        finally:
            conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # Total companies
        cursor.execute('SELECT COUNT(*) FROM companies')
        total_companies = cursor.fetchone()[0]
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # Average credit score
        cursor.execute('SELECT AVG(credit_score) FROM predictions')
        avg_credit_score = cursor.fetchone()[0] or 0
        
        # Risk distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM predictions
            GROUP BY risk_level
        ''')
        risk_distribution = dict(cursor.fetchall())
        
        # Rating distribution
        cursor.execute('''
            SELECT predicted_rating, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_rating
        ''')
        rating_distribution = dict(cursor.fetchall())
        
        # Graph statistics
        cursor.execute('SELECT COUNT(*) FROM graph_edges')
        total_edges = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(similarity_score) FROM graph_edges')
        avg_similarity = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_companies': total_companies,
            'total_predictions': total_predictions,
            'average_credit_score': round(avg_credit_score, 2),
            'risk_distribution': risk_distribution,
            'rating_distribution': rating_distribution,
            'graph_edges': total_edges,
            'average_similarity': round(avg_similarity, 4)
        }


# Create global database instance
db = CreditRiskDatabase()