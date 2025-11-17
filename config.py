"""
Configuration File for Credit Risk Modeling System
Centralized configuration for easy customization
"""

import os


class Config:
    """Base configuration"""
    
    # Application Settings
    APP_NAME = "Credit Risk Modeling with Graph Machine Learning"
    APP_VERSION = "1.0.0"
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Database Settings
    DATABASE_NAME = 'credit_risk_gnn.db'
    DATABASE_PATH = os.path.join(os.path.dirname(__file__), DATABASE_NAME)
    
    # Dataset Settings
    DATASET_PATH = 'corporate_credit_rating.csv'
    DATASET_SOURCE = 'Kaggle Corporate Credit Rating'
    DATASET_URL = 'https://www.kaggle.com/agewerc/corporate-credit-rating'
    
    # GNN Model Settings
    GNN_HIDDEN_DIM = 32  # As per Das et al. (2023)
    GNN_NUM_LAYERS = 2
    GNN_DROPOUT = 0.5
    GNN_NUM_CLASSES = 2  # Binary: Investment Grade vs Below
    
    # Training Settings
    LEARNING_RATE = 0.02  # As per paper
    WEIGHT_DECAY = 0.000294  # As per paper
    EPOCHS = 120  # As per paper
    EARLY_STOPPING_PATIENCE = 20
    BATCH_SIZE = 32
    
    # Graph Construction Settings
    SIMILARITY_THRESHOLD = 0.5  # Range: 0.5-0.7 per paper
    GRAPH_TYPE = 'undirected'
    USE_EDGE_WEIGHTS = False
    
    # Feature Settings
    NUM_FINANCIAL_FEATURES = 25
    USE_GRAPH_FEATURES = True  # Add centrality, clustering
    FEATURE_SCALING = 'standard'  # 'standard', 'minmax', 'robust'
    
    # Prediction Settings
    CREDIT_SCORE_MIN = 300
    CREDIT_SCORE_MAX = 900
    INVESTMENT_GRADE_THRESHOLD = 700
    
    # Rating Thresholds
    RATING_THRESHOLDS = {
        'AAA': 850,
        'AA': 800,
        'A': 750,
        'BBB': 700,
        'BB': 650,
        'B': 600,
        'CCC': 500,
        'D': 0
    }
    
    # Risk Level Thresholds
    RISK_THRESHOLDS = {
        'Low': 750,
        'Medium': 650,
        'High': 550,
        'Very High': 0
    }
    
    # API Settings
    API_PREFIX = '/api'
    CORS_ORIGINS = '*'  # For production, specify allowed origins
    API_RATE_LIMIT = 100  # Requests per minute
    
    # Model Storage
    MODEL_SAVE_DIR = 'saved_models'
    MODEL_CHECKPOINT_FREQ = 10  # Save every N epochs
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'credit_risk_gnn.log'
    
    # Security (for production)
    SECRET_KEY = 'dev-secret-key-change-in-production'
    PASSWORD_HASH_METHOD = 'sha256'
    SESSION_TIMEOUT = 3600  # seconds
    
    # Performance
    MAX_WORKERS = 4  # For parallel processing
    CACHE_ENABLED = True
    CACHE_TTL = 300  # seconds
    
    # Feature Categories (as per Das et al. 2023)
    FEATURE_CATEGORIES = {
        'liquidity': [
            'currentRatio',
            'quickRatio',
            'cashRatio',
            'daysOfSalesOutstanding'
        ],
        'profitability': [
            'grossProfitMargin',
            'operatingProfitMargin',
            'pretaxProfitMargin',
            'netProfitMargin',
            'effectiveTaxRate',
            'returnOnAssets',
            'returnOnEquity',
            'returnOnCapitalEmployed',
            'ebitToRevenue'
        ],
        'debt': [
            'debtRatio',
            'debtEquityRatio'
        ],
        'operating_performance': [
            'assetTurnover',
            'fixedAssetTurnover',
            'companyEquityMultiplier',
            'enterpriseValueMultiple',
            'payableTurnover'
        ],
        'cash_flow': [
            'operatingCashFlowPerShare',
            'freeCashFlowPerShare',
            'cashPerShare',
            'operatingCashFlowSalesRatio',
            'freeCashFlowOperatingCashFlowRatio'
        ]
    }
    
    # Model Performance Targets (from Das et al. 2023)
    TARGET_METRICS = {
        'tabular_only': {
            'f1_score': 0.731,
            'accuracy': 0.682,
            'roc_auc': 0.749,
            'mcc': 0.357
        },
        'tabular_with_graph': {
            'f1_score': 0.755,
            'accuracy': 0.732,
            'roc_auc': 0.808,
            'mcc': 0.459
        },
        'gnn_transductive': {
            'f1_score': 0.826,
            'accuracy': 0.806,
            'roc_auc': 0.866,
            'mcc': 0.610
        },
        'gnn_inductive': {
            'f1_score': 0.807,
            'accuracy': 0.778,
            'roc_auc': 0.863,
            'mcc': 0.555
        },
        'ensemble': {
            'f1_score': 0.817,
            'accuracy': 0.796,
            'roc_auc': 0.862,
            'mcc': 0.588
        }
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Override security settings for production
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'production-secret-key'
    
    # Database
    DATABASE_PATH = os.environ.get('DATABASE_PATH') or Config.DATABASE_PATH
    
    # API
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    
    # Performance
    MAX_WORKERS = int(os.environ.get('MAX_WORKERS', 8))


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_NAME = 'credit_risk_gnn_test.db'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


# Get configuration based on environment
def get_config(env=None):
    """
    Get configuration based on environment
    
    Args:
        env: Environment name (development, production, testing)
        
    Returns:
        Configuration class
    """
    if env is None:
        env = os.environ.get('FLASK_ENV', 'development')
    
    return config.get(env, config['default'])


# Utility functions
def get_all_features():
    """Get list of all financial features"""
    features = []
    for category in Config.FEATURE_CATEGORIES.values():
        features.extend(category)
    return features


def get_rating_from_score(score):
    """Convert credit score to rating"""
    for rating, threshold in sorted(
        Config.RATING_THRESHOLDS.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        if score >= threshold:
            return rating
    return 'D'


def get_risk_level_from_score(score):
    """Convert credit score to risk level"""
    for risk, threshold in sorted(
        Config.RISK_THRESHOLDS.items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        if score >= threshold:
            return risk
    return 'Very High'


def is_investment_grade(score):
    """Check if score qualifies as investment grade"""
    return score >= Config.INVESTMENT_GRADE_THRESHOLD


# Export commonly used config
DEFAULT_CONFIG = get_config()