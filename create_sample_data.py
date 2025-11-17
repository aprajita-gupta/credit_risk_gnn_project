"""
Sample Data Generator for Testing
Creates synthetic company financial data when Kaggle dataset is not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_dataset(n_companies=100, output_file='corporate_credit_rating.csv'):
    """
    Generate synthetic company financial data for testing
    
    Args:
        n_companies: Number of companies to generate
        output_file: Output CSV filename
    """
    np.random.seed(42)
    
    print(f"🔧 Generating sample dataset with {n_companies} companies...")
    
    # Company names
    company_names = [f"Sample Corp {i+1}" for i in range(n_companies)]
    symbols = [f"SC{i:03d}" for i in range(n_companies)]
    
    # Rating distribution (similar to real dataset)
    ratings = []
    rating_dist = {
        'AAA': 0.05, 'AA+': 0.05, 'AA': 0.08, 'AA-': 0.07,
        'A+': 0.10, 'A': 0.15, 'A-': 0.12,
        'BBB+': 0.10, 'BBB': 0.12, 'BBB-': 0.08,
        'BB+': 0.04, 'BB': 0.03, 'BB-': 0.01
    }
    
    for rating, prob in rating_dist.items():
        count = int(n_companies * prob)
        ratings.extend([rating] * count)
    
    # Fill remaining with random ratings
    while len(ratings) < n_companies:
        ratings.append(np.random.choice(list(rating_dist.keys())))
    
    np.random.shuffle(ratings)
    
    # Industry sectors
    sectors = ['Technology', 'Financial', 'Healthcare', 'Consumer', 'Industrial', 
               'Energy', 'Materials', 'Utilities', 'Real Estate', 'Communication']
    company_sectors = np.random.choice(sectors, n_companies)
    
    # Generate financial ratios
    data = {
        'Name': company_names,
        'Symbol': symbols,
        'Rating': ratings,
        'Sector': company_sectors,
        'Rating Agency': np.random.choice(['S&P', 'Moody\'s', 'Fitch'], n_companies),
        'Date': [(datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d') 
                 for _ in range(n_companies)]
    }
    
    # Liquidity Ratios
    data['currentRatio'] = np.random.uniform(0.8, 3.0, n_companies)
    data['quickRatio'] = data['currentRatio'] * np.random.uniform(0.6, 0.9, n_companies)
    data['cashRatio'] = data['quickRatio'] * np.random.uniform(0.3, 0.8, n_companies)
    data['daysOfSalesOutstanding'] = np.random.uniform(30, 90, n_companies)
    
    # Profitability Ratios
    data['grossProfitMargin'] = np.random.uniform(0.15, 0.60, n_companies)
    data['operatingProfitMargin'] = data['grossProfitMargin'] * np.random.uniform(0.3, 0.8, n_companies)
    data['pretaxProfitMargin'] = data['operatingProfitMargin'] * np.random.uniform(0.7, 1.0, n_companies)
    data['netProfitMargin'] = data['pretaxProfitMargin'] * np.random.uniform(0.6, 0.9, n_companies)
    data['effectiveTaxRate'] = np.random.uniform(0.15, 0.35, n_companies)
    data['returnOnAssets'] = np.random.uniform(0.01, 0.15, n_companies)
    data['returnOnEquity'] = np.random.uniform(0.05, 0.25, n_companies)
    data['returnOnCapitalEmployed'] = np.random.uniform(0.05, 0.20, n_companies)
    data['ebitToRevenue'] = data['operatingProfitMargin'] * np.random.uniform(0.9, 1.1, n_companies)
    
    # Debt Ratios
    data['debtRatio'] = np.random.uniform(0.20, 0.80, n_companies)
    data['debtEquityRatio'] = np.random.uniform(0.30, 2.50, n_companies)
    
    # Operating Performance Ratios
    data['assetTurnover'] = np.random.uniform(0.3, 2.0, n_companies)
    data['fixedAssetTurnover'] = data['assetTurnover'] * np.random.uniform(1.0, 3.0, n_companies)
    data['companyEquityMultiplier'] = 1 + data['debtEquityRatio']
    data['enterpriseValueMultiple'] = np.random.uniform(5, 20, n_companies)
    data['payableTurnover'] = np.random.uniform(4, 12, n_companies)
    
    # Cash Flow Ratios
    data['operatingCashFlowPerShare'] = np.random.uniform(1, 10, n_companies)
    data['freeCashFlowPerShare'] = data['operatingCashFlowPerShare'] * np.random.uniform(0.4, 0.9, n_companies)
    data['cashPerShare'] = np.random.uniform(2, 15, n_companies)
    data['operatingCashFlowSalesRatio'] = np.random.uniform(0.05, 0.25, n_companies)
    data['freeCashFlowOperatingCashFlowRatio'] = np.random.uniform(0.4, 0.9, n_companies)
    
    # Adjust ratios based on rating (better ratings = better ratios)
    rating_scores = {
        'AAA': 1.5, 'AA+': 1.4, 'AA': 1.3, 'AA-': 1.2,
        'A+': 1.1, 'A': 1.0, 'A-': 0.9,
        'BBB+': 0.8, 'BBB': 0.7, 'BBB-': 0.6,
        'BB+': 0.5, 'BB': 0.4, 'BB-': 0.3
    }
    
    for i, rating in enumerate(ratings):
        score = rating_scores.get(rating, 0.5)
        # Boost profitability for higher ratings
        data['returnOnAssets'][i] *= score
        data['returnOnEquity'][i] *= score
        # Lower debt for higher ratings
        data['debtRatio'][i] *= (2 - score)
        data['debtEquityRatio'][i] *= (2 - score)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some missing values (realistic)
    for col in df.select_dtypes(include=[np.number]).columns:
        mask = np.random.random(n_companies) < 0.02  # 2% missing
        df.loc[mask, col] = np.nan
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"✅ Sample dataset created: {output_file}")
    print(f"   Companies: {n_companies}")
    print(f"   Features: {len(df.columns)}")
    print(f"   Ratings distribution:")
    print(df['Rating'].value_counts().head(10))
    print(f"\n📊 Sample statistics:")
    print(f"   Avg ROA: {df['returnOnAssets'].mean():.4f}")
    print(f"   Avg Debt Ratio: {df['debtRatio'].mean():.4f}")
    print(f"   Avg Current Ratio: {df['currentRatio'].mean():.4f}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample credit rating dataset')
    parser.add_argument('--companies', type=int, default=100,
                       help='Number of companies to generate')
    parser.add_argument('--output', type=str, default='corporate_credit_rating.csv',
                       help='Output CSV filename')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 Sample Credit Rating Dataset Generator")
    print("=" * 80)
    
    df = generate_sample_dataset(args.companies, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Dataset generation complete!")
    print(f"   File: {args.output}")
    print(f"   Ready to use with: python app.py")
    print("=" * 80)