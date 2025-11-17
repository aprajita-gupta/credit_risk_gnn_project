"""
Data Loader for Corporate Credit Rating Dataset
Based on Das et al. (2023) - Uses Kaggle dataset with 2,029 ratings and 25 financial features
Dataset: https://www.kaggle.com/agewerc/corporate-credit-rating
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


class CreditRatingDataLoader:
    """
    Loads and processes the Corporate Credit Rating dataset from Kaggle
    Features: 25 financial ratios (liquidity, profitability, debt, operating performance, cash flow)
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize data loader. Will auto-detect common CSV filenames if not specified.
        
        Args:
            data_path: Path to CSV file. If None, searches for common names.
        """
        # Auto-detect dataset file if not specified
        if data_path is None:
            possible_files = [
                'corporate_credit_rating.csv',
                'corporate_rating.csv',  # User's filename
                'credit_rating.csv',
                'data.csv'
            ]
            
            for filename in possible_files:
                if os.path.exists(filename):
                    data_path = filename
                    print(f"✅ Found dataset: {filename}")
                    break
            
            if data_path is None:
                raise FileNotFoundError(
                    "❌ Dataset not found! Please place your CSV file here.\n"
                    "   Expected names: corporate_credit_rating.csv or corporate_rating.csv"
                )
        
        self.data_path = data_path
        self.df = None
        self.feature_columns = []
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Define the 25 financial ratio features as per the paper
        self.ratio_features = {
            'liquidity': ['currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding'],
            'profitability': ['grossProfitMargin', 'operatingProfitMargin', 'pretaxProfitMargin',
                            'netProfitMargin', 'effectiveTaxRate', 'returnOnAssets',
                            'returnOnEquity', 'returnOnCapitalEmployed', 'ebitToRevenue'],
            'debt': ['debtRatio', 'debtEquityRatio'],
            'operating_performance': ['assetTurnover', 'fixedAssetTurnover', 
                                     'companyEquityMultiplier', 'enterpriseValueMultiple',
                                     'payableTurnover'],
            'cash_flow': ['operatingCashFlowPerShare', 'freeCashFlowPerShare',
                         'cashPerShare', 'operatingCashFlowSalesRatio',
                         'freeCashFlowOperatingCashFlowRatio']
        }
        
        self.load_data()
    
    def load_data(self):
        """Load the dataset"""
        if not os.path.exists(self.data_path):
            print(f"⚠️ Dataset not found at {self.data_path}")
            print("Please download from: https://www.kaggle.com/agewerc/corporate-credit-rating")
            return None
        
        print(f"📊 Loading Corporate Credit Rating dataset from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Get all feature columns (flatten the ratio_features dict)
        for category in self.ratio_features.values():
            self.feature_columns.extend(category)
        
        print(f"✅ Loaded {len(self.df)} records with {len(self.feature_columns)} financial ratio features")
        print(f"📈 Rating distribution:")
        if 'Rating' in self.df.columns:
            print(self.df['Rating'].value_counts())
        
        return self.df
    
    def create_binary_labels(self) -> pd.Series:
        """
        Create binary labels: Investment Grade (1) vs Below Investment Grade (0)
        Investment Grade: AAA, AA, A, BBB
        Below Investment Grade: BB, B, CCC, CC, C, D
        """
        if self.df is None or 'Rating' not in self.df.columns:
            return None
        
        investment_grade = ['AAA', 'AA', 'A', 'BBB', 'AA+', 'AA-', 'A+', 'A-', 'BBB+', 'BBB-']
        
        self.df['InvestmentGrade'] = self.df['Rating'].apply(
            lambda x: 1 if any(ig in str(x) for ig in investment_grade) else 0
        )
        
        ig_count = (self.df['InvestmentGrade'] == 1).sum()
        big_count = (self.df['InvestmentGrade'] == 0).sum()
        
        print(f"\n📊 Binary Classification Labels:")
        print(f"   Investment Grade (1): {ig_count} ({ig_count/len(self.df)*100:.1f}%)")
        print(f"   Below Investment Grade (0): {big_count} ({big_count/len(self.df)*100:.1f}%)")
        
        return self.df['InvestmentGrade']
    
    def get_features_and_labels(self, add_graph_features: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get feature matrix and labels
        
        Args:
            add_graph_features: If True, adds degree_centrality, eigenvector_centrality, 
                              clustering_coefficient columns (to be filled later)
        
        Returns:
            X: Feature matrix
            y: Binary labels (1=Investment Grade, 0=Below Investment Grade)
        """
        if self.df is None:
            return None, None
        
        # Create binary labels
        y = self.create_binary_labels()
        
        # Get feature columns that exist in the dataframe
        available_features = [col for col in self.feature_columns if col in self.df.columns]
        
        if len(available_features) < len(self.feature_columns):
            print(f"⚠️ Warning: Only {len(available_features)}/{len(self.feature_columns)} features found")
            print(f"   Missing features: {set(self.feature_columns) - set(available_features)}")
        
        X = self.df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Add placeholder graph features if requested
        if add_graph_features:
            X['degree_centrality'] = 0.0
            X['eigenvector_centrality'] = 0.0
            X['clustering_coefficient'] = 0.0
        
        print(f"\n✅ Feature matrix shape: {X.shape}")
        print(f"   Features: {len(X.columns)}")
        print(f"   Samples: {len(X)}")
        
        return X, y
    
    def prepare_train_test_split(self, test_size: float = 0.2, random_state: int = 42,
                                add_graph_features: bool = False) -> Dict:
        """
        Prepare train/test split
        
        Returns:
            Dictionary with X_train, X_test, y_train, y_test, feature_names
        """
        X, y = self.get_features_and_labels(add_graph_features=add_graph_features)
        
        if X is None or y is None:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit scaler on training data only
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"\n📊 Train/Test Split:")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Train IG ratio: {y_train.mean():.2%}")
        print(f"   Test IG ratio: {y_test.mean():.2%}")
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': X.columns.tolist(),
            'company_names': self.df.loc[X.index, 'Name'].values if 'Name' in self.df.columns else None,
            'original_ratings': self.df.loc[X.index, 'Rating'].values if 'Rating' in self.df.columns else None
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if self.df is None:
            return {}
        
        stats = {
            'total_companies': len(self.df),
            'total_features': len(self.feature_columns)
        }
        
        # Calculate statistics for each ratio category
        for category, features in self.ratio_features.items():
            available = [f for f in features if f in self.df.columns]
            if available:
                stats[f'{category}_features'] = len(available)
                stats[f'avg_{category}'] = self.df[available].mean().mean()
        
        # Rating statistics
        if 'Rating' in self.df.columns:
            stats['rating_distribution'] = self.df['Rating'].value_counts().to_dict()
            
            # Investment grade ratio
            investment_grade = ['AAA', 'AA', 'A', 'BBB', 'AA+', 'AA-', 'A+', 'A-', 'BBB+', 'BBB-']
            ig_count = self.df['Rating'].apply(
                lambda x: any(ig in str(x) for ig in investment_grade)
            ).sum()
            stats['investment_grade_ratio'] = ig_count / len(self.df)
        
        return stats
    
    def get_company_data(self, company_name: str = None, company_id: int = None) -> Optional[Dict]:
        """Get data for a specific company"""
        if self.df is None:
            return None
        
        if company_name and 'Name' in self.df.columns:
            company = self.df[self.df['Name'] == company_name]
        elif company_id is not None:
            company = self.df.iloc[company_id:company_id+1]
        else:
            return None
        
        if len(company) == 0:
            return None
        
        return company.iloc[0].to_dict()
    
    def predict_credit_score_simple(self, features: Dict[str, float]) -> float:
        """
        Simple credit score prediction based on financial ratios
        Returns score between 300-900
        
        Balanced formula based on typical financial ratio ranges:
        - Excellent companies: 750-900
        - Good companies: 650-750
        - Average companies: 550-650
        - Below average: 450-550
        - Poor: 300-450
        """
        # Start with middle score
        score = 550
        
        # Profitability factors (typically 0-0.20 or 0-20%)
        if 'returnOnAssets' in features:
            roa = features['returnOnAssets']
            # Good ROA: >10% (+50), Average: 5% (0), Poor: <2% (-30)
            score += (roa - 0.05) * 500  # Range: -25 to +75
        
        if 'returnOnEquity' in features:
            roe = features['returnOnEquity']
            # Good ROE: >15% (+30), Average: 10% (0), Poor: <5% (-20)
            score += (roe - 0.10) * 300  # Range: -30 to +60
        
        if 'netProfitMargin' in features:
            npm = features['netProfitMargin']
            # Good margin: >10% (+20), Average: 5% (0), Poor: <2% (-15)
            score += (npm - 0.05) * 200  # Range: -20 to +40
        
        # Liquidity factors
        if 'currentRatio' in features:
            cr = features['currentRatio']
            # Good: >2.0 (+30), Adequate: 1.5 (0), Poor: <1.0 (-40)
            if cr >= 1.5:
                score += min((cr - 1.5) * 60, 60)  # Max +60 for excellent liquidity
            else:
                score -= (1.5 - cr) * 80  # Penalty for low liquidity
        
        if 'quickRatio' in features:
            qr = features['quickRatio']
            # Good: >1.0 (+15), Adequate: 0.8 (0), Poor: <0.5 (-20)
            score += (qr - 0.8) * 75  # Range: -22.5 to +15
        
        # Debt factors (PENALTIES)
        if 'debtRatio' in features:
            dr = features['debtRatio']
            # Low debt: <0.3 (+20), Average: 0.5 (0), High: >0.7 (-60)
            score -= (dr - 0.3) * 300  # Heavy penalty for high debt
        
        if 'debtEquityRatio' in features:
            der = features['debtEquityRatio']
            # Low: <0.5 (+10), Average: 1.0 (0), High: >2.0 (-50)
            score -= (der - 1.0) * 50  # Penalty for high leverage
        
        # Operating efficiency
        if 'operatingProfitMargin' in features:
            opm = features['operatingProfitMargin']
            # Good: >15% (+20), Average: 10% (0), Poor: <5% (-15)
            score += (opm - 0.10) * 150  # Range: -15 to +30
        
        # Cash flow strength
        if 'operatingCashFlowSalesRatio' in features:
            ocf = features['operatingCashFlowSalesRatio']
            # Strong: >15% (+25), Average: 10% (0), Weak: <5% (-15)
            score += (ocf - 0.10) * 250  # Range: -25 to +50
        
        # Asset efficiency
        if 'assetTurnover' in features:
            at = features['assetTurnover']
            # Efficient: >1.5 (+15), Average: 1.0 (0), Inefficient: <0.5 (-10)
            score += (at - 1.0) * 30  # Range: -15 to +15
        
        # Clamp to valid range (300-900)
        score = max(300, min(900, score))
        
        return round(score, 2)
    
    def get_risk_level(self, credit_score: float) -> str:
        """Determine risk level based on credit score"""
        if credit_score >= 750:
            return 'Low'
        elif credit_score >= 650:
            return 'Medium'
        elif credit_score >= 550:
            return 'High'
        else:
            return 'Very High'
    
    def get_predicted_rating(self, credit_score: float) -> str:
        """Convert credit score to rating category"""
        if credit_score >= 850:
            return 'AAA'
        elif credit_score >= 800:
            return 'AA'
        elif credit_score >= 750:
            return 'A'
        elif credit_score >= 700:
            return 'BBB'
        elif credit_score >= 650:
            return 'BB'
        elif credit_score >= 600:
            return 'B'
        elif credit_score >= 500:
            return 'CCC'
        else:
            return 'D'


# Create global data loader instance
data_loader = CreditRatingDataLoader()