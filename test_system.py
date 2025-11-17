"""
Test Script for Credit Risk Modeling System
Tests all major components of the application
"""

import requests
import json
import time
from typing import Dict, List


class CreditRiskTester:
    """Test suite for credit risk modeling system"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def test(self, name: str, func):
        """Run a test and record results"""
        print(f"\n🧪 Testing: {name}")
        try:
            func()
            print(f"   ✅ PASSED")
            self.passed += 1
            self.results.append({'test': name, 'status': 'PASSED'})
        except AssertionError as e:
            print(f"   ❌ FAILED: {e}")
            self.failed += 1
            self.results.append({'test': name, 'status': 'FAILED', 'error': str(e)})
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            self.failed += 1
            self.results.append({'test': name, 'status': 'ERROR', 'error': str(e)})
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.api_url}/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'status' in data, "Response missing 'status' field"
        assert data['status'] == 'healthy', f"System not healthy: {data.get('status')}"
        print(f"      System status: {data['status']}")
        print(f"      Total companies: {data.get('total_companies', 'N/A')}")
    
    def test_get_companies(self):
        """Test getting company list"""
        response = requests.get(f"{self.api_url}/companies?limit=10")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'companies' in data, "Response missing 'companies' field"
        assert isinstance(data['companies'], list), "Companies should be a list"
        print(f"      Retrieved {len(data['companies'])} companies")
    
    def test_register_user(self):
        """Test user registration"""
        user_data = {
            'email': f'test{int(time.time())}@example.com',
            'password': 'testpass123',
            'name': 'Test User'
        }
        
        response = requests.post(f"{self.api_url}/register", json=user_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data.get('success') == True, "Registration failed"
        print(f"      User registered: {data['user']['email']}")
    
    def test_predict_simple(self):
        """Test simple credit risk prediction"""
        prediction_data = {
            'company_name': 'Test Corporation',
            'returnOnAssets': 0.06,
            'debtRatio': 0.40,
            'currentRatio': 1.8,
            'debtEquityRatio': 0.75
        }
        
        response = requests.post(f"{self.api_url}/predict", json=prediction_data)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert data.get('success') == True, "Prediction failed"
        assert 'credit_score' in data, "Missing credit_score"
        assert 'predicted_rating' in data, "Missing predicted_rating"
        assert 'risk_level' in data, "Missing risk_level"
        
        print(f"      Credit Score: {data['credit_score']}")
        print(f"      Rating: {data['predicted_rating']}")
        print(f"      Risk Level: {data['risk_level']}")
        print(f"      Investment Grade: {data.get('investment_grade')}")
    
    def test_predict_with_company_id(self):
        """Test prediction with company ID"""
        # First get a company
        response = requests.get(f"{self.api_url}/companies?limit=1")
        companies = response.json()['companies']
        
        if len(companies) > 0:
            company_id = companies[0]['id']
            
            prediction_data = {'company_id': company_id}
            response = requests.post(f"{self.api_url}/predict", json=prediction_data)
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data.get('success') == True, "Prediction failed"
            print(f"      Predicted for: {data.get('company_name')}")
            print(f"      Credit Score: {data['credit_score']}")
        else:
            print("      ⚠️  No companies available, skipping")
    
    def test_get_statistics(self):
        """Test statistics endpoint"""
        response = requests.get(f"{self.api_url}/statistics")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'dataset' in data, "Missing dataset statistics"
        assert 'predictions' in data, "Missing prediction statistics"
        
        print(f"      Total companies: {data['dataset'].get('total_companies')}")
        print(f"      Total predictions: {data['predictions'].get('total_predictions')}")
        print(f"      Avg credit score: {data['predictions'].get('average_credit_score')}")
    
    def test_dataset_info(self):
        """Test dataset info endpoint"""
        response = requests.get(f"{self.api_url}/dataset/info")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'source' in data, "Missing source info"
        assert 'features' in data, "Missing features info"
        assert 'methodology' in data, "Missing methodology info"
        
        print(f"      Dataset: {data['source']}")
        print(f"      Features: {len(data['features'])} categories")
    
    def test_predictions_history(self):
        """Test getting predictions history"""
        response = requests.get(f"{self.api_url}/predictions?limit=5")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
        data = response.json()
        assert 'predictions' in data, "Missing predictions"
        assert 'count' in data, "Missing count"
        
        print(f"      Retrieved {data['count']} predictions")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with missing data
        response = requests.post(f"{self.api_url}/predict", json={})
        assert response.status_code in [400, 500], "Should fail with empty data"
        
        # Test with invalid company ID
        response = requests.post(f"{self.api_url}/predict", 
                                json={'company_id': 999999})
        assert response.status_code in [404, 400], "Should fail with invalid ID"
        
        print(f"      Edge cases handled correctly")
    
    def test_data_validation(self):
        """Test data validation"""
        # Test with extreme values
        prediction_data = {
            'company_name': 'Test Corp',
            'returnOnAssets': 10.0,  # Extremely high
            'debtRatio': -0.5,  # Negative (invalid)
            'currentRatio': 1.5
        }
        
        response = requests.post(f"{self.api_url}/predict", json=prediction_data)
        # Should either succeed with clamped values or return validation error
        assert response.status_code in [200, 400], "Should handle extreme values"
        
        print(f"      Data validation working")
    
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import concurrent.futures
        
        def make_prediction():
            data = {
                'company_name': 'Concurrent Test',
                'returnOnAssets': 0.05,
                'debtRatio': 0.45,
                'currentRatio': 1.6
            }
            response = requests.post(f"{self.api_url}/predict", json=data)
            return response.status_code == 200
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_prediction) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success_rate = sum(results) / len(results) * 100
        assert success_rate >= 80, f"Too many failures: {success_rate}% success"
        
        print(f"      Concurrent requests: {success_rate}% success rate")
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 80)
        print("🚀 Credit Risk Modeling System - Test Suite")
        print("=" * 80)
        
        # Connection test
        print("\n📡 Testing API connection...")
        try:
            response = requests.get(self.base_url, timeout=5)
            print(f"   ✅ Server is running at {self.base_url}")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to {self.base_url}")
            print("   Make sure the server is running: python app.py")
            return
        
        # Run all tests
        self.test("Health Check", self.test_health_check)
        self.test("Get Companies", self.test_get_companies)
        self.test("User Registration", self.test_register_user)
        self.test("Simple Prediction", self.test_predict_simple)
        self.test("Prediction with Company ID", self.test_predict_with_company_id)
        self.test("Get Statistics", self.test_get_statistics)
        self.test("Dataset Info", self.test_dataset_info)
        self.test("Predictions History", self.test_predictions_history)
        self.test("Edge Cases", self.test_edge_cases)
        self.test("Data Validation", self.test_data_validation)
        self.test("Concurrent Requests", self.test_concurrent_requests)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 Test Summary")
        print("=" * 80)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success Rate: {self.passed / (self.passed + self.failed) * 100:.1f}%")
        print("=" * 80)
        
        # Detailed results
        if self.failed > 0:
            print("\n📋 Failed Tests:")
            for result in self.results:
                if result['status'] != 'PASSED':
                    print(f"   ❌ {result['test']}: {result.get('error', 'Unknown error')}")
        
        return self.failed == 0


def run_performance_test(base_url: str = "http://localhost:5000", 
                        num_requests: int = 100):
    """
    Run performance test
    
    Args:
        base_url: Base URL of the application
        num_requests: Number of requests to make
    """
    print("\n" + "=" * 80)
    print(f"⚡ Performance Test - {num_requests} Requests")
    print("=" * 80)
    
    api_url = f"{base_url}/api"
    prediction_data = {
        'company_name': 'Performance Test',
        'returnOnAssets': 0.06,
        'debtRatio': 0.40,
        'currentRatio': 1.8
    }
    
    start_time = time.time()
    successful = 0
    failed = 0
    response_times = []
    
    for i in range(num_requests):
        req_start = time.time()
        try:
            response = requests.post(f"{api_url}/predict", json=prediction_data, timeout=10)
            req_time = time.time() - req_start
            response_times.append(req_time)
            
            if response.status_code == 200:
                successful += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"\n   Error on request {i+1}: {e}")
        
        if (i + 1) % 20 == 0:
            print(f"   Progress: {i+1}/{num_requests} requests completed")
    
    total_time = time.time() - start_time
    
    # Results
    import numpy as np
    print("\n📊 Performance Results:")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Requests/Second: {num_requests / total_time:.2f}")
    print(f"   Successful: {successful}/{num_requests} ({successful/num_requests*100:.1f}%)")
    print(f"   Failed: {failed}/{num_requests}")
    
    if response_times:
        print(f"\n⏱️  Response Times:")
        print(f"   Mean: {np.mean(response_times)*1000:.0f}ms")
        print(f"   Median: {np.median(response_times)*1000:.0f}ms")
        print(f"   Min: {np.min(response_times)*1000:.0f}ms")
        print(f"   Max: {np.max(response_times)*1000:.0f}ms")
        print(f"   95th percentile: {np.percentile(response_times, 95)*1000:.0f}ms")
    
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Credit Risk Modeling System')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                       help='Base URL of the application')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance test')
    parser.add_argument('--requests', type=int, default=100,
                       help='Number of requests for performance test')
    
    args = parser.parse_args()
    
    # Run functional tests
    tester = CreditRiskTester(args.url)
    all_passed = tester.run_all_tests()
    
    # Run performance test if requested
    if args.performance:
        run_performance_test(args.url, args.requests)
    
    # Exit code
    exit(0 if all_passed else 1)