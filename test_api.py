# test_api.py
"""
Test script for the Restaurant Rating Predictor API
"""

import requests
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url="http://localhost:8005"):
        self.base_url = base_url
        
    def wait_for_api(self, timeout=60):
        """Wait for API to be ready"""
        logger.info(f"Waiting for API at {self.base_url}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ API is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.error("❌ API is not responding")
        return False
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        logger.info("Testing root endpoint...")
        try:
            response = requests.get(f"{self.base_url}/")
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Root endpoint working")
            logger.info(f"Response: {data['message']}")
            return True
        except Exception as e:
            logger.error(f"❌ Root endpoint failed: {str(e)}")
            return False
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        logger.info("Testing health endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Health endpoint working")
            logger.info(f"Status: {data['status']}")
            logger.info(f"Model loaded: {data['model_loaded']}")
            return True
        except Exception as e:
            logger.error(f"❌ Health endpoint failed: {str(e)}")
            return False
    
    def test_prediction_endpoint(self):
        """Test the single prediction endpoint"""
        logger.info("Testing prediction endpoint...")
        
        test_data = {
            "name": "Test Restaurant",
            "location": "Downtown",
            "categories": "Italian",
            "price_range": "$$",
            "text": "Amazing food and excellent service! The pasta was perfectly cooked and the atmosphere was wonderful. Highly recommend this place for a romantic dinner."
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Prediction endpoint working")
            logger.info(f"Predicted rating: {data['predicted_rating']} stars")
            logger.info(f"Confidence: {data['confidence']}")
            logger.info(f"Sentiment: {data['review_sentiment']}")
            return True
        except Exception as e:
            logger.error(f"❌ Prediction endpoint failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def test_batch_prediction_endpoint(self):
        """Test the batch prediction endpoint"""
        logger.info("Testing batch prediction endpoint...")
        
        test_data = {
            "restaurants": [
                {
                    "name": "Great Pizza Place",
                    "location": "Downtown",
                    "categories": "Italian",
                    "price_range": "$$",
                    "text": "Fantastic pizza with fresh ingredients! The service was quick and friendly. Definitely coming back!"
                },
                {
                    "name": "Terrible Burger Joint",
                    "location": "Suburb",
                    "categories": "American",
                    "price_range": "$",
                    "text": "Worst food ever. The burger was cold and the fries were soggy. Terrible service and dirty restaurant."
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/predict_batch",
                json=test_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Batch prediction endpoint working")
            logger.info(f"Total processed: {data['total_processed']}")
            
            for i, prediction in enumerate(data['predictions']):
                logger.info(f"Restaurant {i+1}: {prediction['predicted_rating']} stars (confidence: {prediction['confidence']})")
            
            return True
        except Exception as e:
            logger.error(f"❌ Batch prediction endpoint failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return False
    
    def test_sample_data_endpoint(self):
        """Test the sample data endpoint"""
        logger.info("Testing sample data endpoint...")
        try:
            response = requests.get(f"{self.base_url}/sample_data")
            response.raise_for_status()
            
            data = response.json()
            logger.info("✅ Sample data endpoint working")
            logger.info(f"Sample restaurants available: {len(data['sample_restaurants'])}")
            return True
        except Exception as e:
            logger.error(f"❌ Sample data endpoint failed: {str(e)}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        logger.info("="*60)
        logger.info("STARTING API TESTS")
        logger.info("="*60)
        
        # Wait for API to be ready
        if not self.wait_for_api():
            logger.error("API is not available. Make sure it's running!")
            return False
        
        tests = [
            ("Root Endpoint", self.test_root_endpoint),
            ("Health Endpoint", self.test_health_endpoint),
            ("Sample Data Endpoint", self.test_sample_data_endpoint),
            ("Prediction Endpoint", self.test_prediction_endpoint),
            ("Batch Prediction Endpoint", self.test_batch_prediction_endpoint)
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_function in tests:
            logger.info(f"\n{'-'*40}")
            logger.info(f"TEST: {test_name}")
            logger.info(f"{'-'*40}")
            
            if test_function():
                passed += 1
            else:
                failed += 1
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        if failed == 0:
            logger.info("🎉 All tests passed!")
        else:
            logger.warning(f"⚠️  {failed} tests failed")
        
        return failed == 0

def main():
    """Main function"""
    print("Restaurant Rating Predictor API Tester")
    print("Make sure the API is running on http://localhost:8005")
    print("You can start it with: python start_api.py")
    print()
    
    # Ask user if API is running
    response = input("Is the API running? (y/n): ")
    if response.lower() != 'y':
        print("Please start the API first and then run this test script again.")
        return
    
    tester = APITester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Your API is working correctly.")
        print("\nYou can now:")
        print("1. Visit http://localhost:8005/docs for interactive API documentation")
        print("2. Use the API endpoints in your applications")
        print("3. Deploy the API to production")
    else:
        print("\n⚠️  Some tests failed. Please check the logs above.")

if __name__ == "__main__":
    main()