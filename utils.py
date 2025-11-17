"""
Utility Functions for Credit Risk Modeling System
Helper functions used across the application
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json


def calculate_credit_score(financial_ratios: Dict[str, float], 
                          weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate credit score from financial ratios
    
    Args:
        financial_ratios: Dictionary of financial ratio values
        weights: Optional custom weights for each ratio
        
    Returns:
        Credit score between 300-900
    """
    # Default weights based on importance
    if weights is None:
        weights = {
            'returnOnAssets': 200,
            'returnOnEquity': 150,
            'currentRatio': 100,
            'debtRatio': -150,
            'debtEquityRatio': -100,
            'operatingProfitMargin': 120,
            'netProfitMargin': 100,
            'operatingCashFlowSalesRatio': 80
        }
    
    # Base score
    score = 500
    
    # Apply weights
    for ratio, weight in weights.items():
        if ratio in financial_ratios:
            value = financial_ratios[ratio]
            if not pd.isna(value):
                score += value * weight
    
    # Clamp to valid range
    score = max(300, min(900, score))
    
    return round(score, 2)


def categorize_rating(rating: str) -> str:
    """
    Categorize rating into Investment Grade or Below Investment Grade
    
    Args:
        rating: Credit rating (AAA, AA, A, BBB, BB, B, CCC, D)
        
    Returns:
        'Investment Grade' or 'Below Investment Grade'
    """
    investment_grade = ['AAA', 'AA', 'A', 'BBB', 'AA+', 'AA-', 'A+', 'A-', 'BBB+', 'BBB-']
    
    for ig_rating in investment_grade:
        if ig_rating in str(rating):
            return 'Investment Grade'
    
    return 'Below Investment Grade'


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency
    
    Args:
        amount: Amount to format
        currency: Currency code (USD, EUR, etc.)
        
    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'EUR':
        return f"€{amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0.5 = 50%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def normalize_features(features: pd.DataFrame, 
                       method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Normalize features using specified method
    
    Args:
        features: Feature DataFrame
        method: 'standard', 'minmax', or 'robust'
        
    Returns:
        Normalized features and scaler object
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    normalized = pd.DataFrame(
        scaler.fit_transform(features),
        columns=features.columns,
        index=features.index
    )
    
    return normalized, scaler


def compute_financial_health_score(ratios: Dict[str, float]) -> Dict[str, float]:
    """
    Compute comprehensive financial health scores
    
    Args:
        ratios: Dictionary of financial ratios
        
    Returns:
        Dictionary with category scores and overall score
    """
    scores = {}
    
    # Liquidity Score (0-100)
    liquidity_score = 0
    if 'currentRatio' in ratios:
        liquidity_score += min(ratios['currentRatio'] / 2 * 50, 50)
    if 'quickRatio' in ratios:
        liquidity_score += min(ratios['quickRatio'] / 1.5 * 50, 50)
    scores['liquidity'] = liquidity_score
    
    # Profitability Score (0-100)
    profitability_score = 0
    if 'returnOnAssets' in ratios:
        profitability_score += min(ratios['returnOnAssets'] * 500, 50)
    if 'returnOnEquity' in ratios:
        profitability_score += min(ratios['returnOnEquity'] * 250, 50)
    scores['profitability'] = profitability_score
    
    # Leverage Score (0-100, lower is better)
    leverage_score = 100
    if 'debtRatio' in ratios:
        leverage_score -= min(ratios['debtRatio'] * 100, 50)
    if 'debtEquityRatio' in ratios:
        leverage_score -= min(ratios['debtEquityRatio'] * 25, 50)
    scores['leverage'] = max(0, leverage_score)
    
    # Overall Score
    scores['overall'] = np.mean(list(scores.values()))
    
    return {k: round(v, 2) for k, v in scores.items()}


def generate_recommendations(ratios: Dict[str, float], 
                            credit_score: float) -> List[str]:
    """
    Generate actionable recommendations based on financial ratios
    
    Args:
        ratios: Dictionary of financial ratios
        credit_score: Current credit score
        
    Returns:
        List of recommendation strings
    """
    recommendations = []
    
    # Debt recommendations
    if ratios.get('debtRatio', 0) > 0.6:
        recommendations.append("🔴 High debt ratio (>60%) - Consider debt reduction strategies")
    elif ratios.get('debtRatio', 0) > 0.4:
        recommendations.append("🟡 Moderate debt ratio - Monitor debt levels closely")
    
    if ratios.get('debtEquityRatio', 0) > 2.0:
        recommendations.append("🔴 High leverage - Reduce debt-to-equity ratio below 2.0")
    
    # Profitability recommendations
    if ratios.get('returnOnAssets', 0) < 0.05:
        recommendations.append("🔴 Low ROA (<5%) - Focus on improving asset utilization")
    
    if ratios.get('returnOnEquity', 0) < 0.10:
        recommendations.append("🔴 Low ROE (<10%) - Work on improving shareholder returns")
    
    if ratios.get('netProfitMargin', 0) < 0.05:
        recommendations.append("🟡 Low profit margin - Consider cost optimization")
    
    # Liquidity recommendations
    if ratios.get('currentRatio', 0) < 1.0:
        recommendations.append("🔴 Current ratio below 1.0 - Immediate liquidity concerns")
    elif ratios.get('currentRatio', 0) < 1.5:
        recommendations.append("🟡 Low current ratio - Improve working capital management")
    
    # Cash flow recommendations
    if ratios.get('operatingCashFlowSalesRatio', 0) < 0.10:
        recommendations.append("🟡 Low operating cash flow - Focus on cash generation")
    
    # Overall score recommendations
    if credit_score < 600:
        recommendations.append("⚠️ Credit score below 600 - Comprehensive financial restructuring needed")
    elif credit_score < 700:
        recommendations.append("🟡 Below investment grade - Strengthen financial position")
    else:
        recommendations.append("✅ Maintain strong financial performance")
    
    # Positive reinforcement
    if ratios.get('returnOnAssets', 0) > 0.10:
        recommendations.append("✅ Strong ROA - Continue efficient asset management")
    
    if ratios.get('currentRatio', 0) > 2.0:
        recommendations.append("✅ Strong liquidity position")
    
    return recommendations[:5]  # Return top 5 recommendations


def calculate_altman_z_score(financial_data: Dict[str, float]) -> float:
    """
    Calculate Altman Z-Score for bankruptcy prediction
    Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    
    Args:
        financial_data: Dictionary with required financial metrics
        
    Returns:
        Altman Z-Score
    """
    # Extract components (would need actual balance sheet data)
    # Simplified version using available ratios
    
    working_capital_to_assets = financial_data.get('currentRatio', 1.5) - 1
    retained_earnings_to_assets = financial_data.get('returnOnAssets', 0.05)
    ebit_to_assets = financial_data.get('operatingProfitMargin', 0.10)
    market_value_to_liabilities = 1 / financial_data.get('debtRatio', 0.5) if financial_data.get('debtRatio', 0) > 0 else 2
    sales_to_assets = financial_data.get('assetTurnover', 1.0)
    
    z_score = (
        1.2 * working_capital_to_assets +
        1.4 * retained_earnings_to_assets +
        3.3 * ebit_to_assets +
        0.6 * market_value_to_liabilities +
        1.0 * sales_to_assets
    )
    
    return round(z_score, 2)


def interpret_altman_z_score(z_score: float) -> Dict[str, str]:
    """
    Interpret Altman Z-Score
    
    Args:
        z_score: Altman Z-Score value
        
    Returns:
        Dictionary with interpretation
    """
    if z_score > 2.99:
        return {
            'zone': 'Safe',
            'risk': 'Low',
            'description': 'Company is financially healthy with low bankruptcy risk'
        }
    elif z_score > 1.81:
        return {
            'zone': 'Grey',
            'risk': 'Medium',
            'description': 'Company is in a grey area - requires monitoring'
        }
    else:
        return {
            'zone': 'Distress',
            'risk': 'High',
            'description': 'Company shows signs of financial distress'
        }


def validate_financial_ratios(ratios: Dict[str, float]) -> Tuple[bool, List[str]]:
    """
    Validate financial ratios for reasonable values
    
    Args:
        ratios: Dictionary of financial ratios
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for negative values where not allowed
    non_negative = ['currentRatio', 'quickRatio', 'assetTurnover']
    for ratio in non_negative:
        if ratio in ratios and ratios[ratio] < 0:
            errors.append(f"{ratio} cannot be negative")
    
    # Check for values outside reasonable ranges
    if 'debtRatio' in ratios:
        if ratios['debtRatio'] < 0 or ratios['debtRatio'] > 1:
            errors.append("debtRatio should be between 0 and 1")
    
    if 'currentRatio' in ratios:
        if ratios['currentRatio'] > 10:
            errors.append("currentRatio seems unusually high (>10)")
    
    # Check for required ratios
    required = ['returnOnAssets', 'debtRatio']
    for ratio in required:
        if ratio not in ratios or pd.isna(ratios[ratio]):
            errors.append(f"Required ratio missing: {ratio}")
    
    return len(errors) == 0, errors


def create_prediction_id() -> str:
    """
    Create unique prediction ID
    
    Returns:
        Unique prediction ID string
    """
    timestamp = int(datetime.now().timestamp())
    return f"PRED{timestamp}"


def create_company_id(name: str) -> str:
    """
    Create company ID from name
    
    Args:
        name: Company name
        
    Returns:
        Company ID string
    """
    # Remove special characters and convert to uppercase
    clean_name = ''.join(c for c in name if c.isalnum() or c.isspace())
    parts = clean_name.split()[:3]  # Take first 3 words
    return 'COMP_' + '_'.join(parts).upper()


def export_to_json(data: Any, filepath: str, indent: int = 2) -> bool:
    """
    Export data to JSON file
    
    Args:
        data: Data to export
        filepath: Output file path
        indent: JSON indentation
        
    Returns:
        Success boolean
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        return True
    except Exception as e:
        print(f"Error exporting to JSON: {e}")
        return False


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistics
    """
    values_array = np.array(values)
    
    return {
        'mean': float(np.mean(values_array)),
        'median': float(np.median(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q25': float(np.percentile(values_array, 25)),
        'q75': float(np.percentile(values_array, 75))
    }


# Logging utility
def log_prediction(prediction_data: Dict, log_file: str = 'predictions.log'):
    """
    Log prediction to file
    
    Args:
        prediction_data: Prediction information
        log_file: Log file path
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"{timestamp} | {prediction_data.get('prediction_id')} | " \
                f"Score: {prediction_data.get('credit_score')} | " \
                f"Rating: {prediction_data.get('predicted_rating')}\n"
    
    try:
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except Exception as e:
        print(f"Error writing to log: {e}")