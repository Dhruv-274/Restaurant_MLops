# run_pipeline.py
"""
Complete MLOps Pipeline Runner
This script runs the entire pipeline from data processing to model training
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed")
        logger.error(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if we're in the right directory
    required_dirs = ['src', 'api', 'data', 'models']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        logger.error(f"Missing directories: {missing_dirs}")
        logger.error("Please make sure you're in the project root directory")
        return False
    
    # Check if Python files exist
    required_files = [
        'src/data_processing.py',
        'src/model_training.py',
        'src/prediction.py',
        'src/main.py'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return False
    
    logger.info("✅ All prerequisites met")
    return True

def setup_nltk():
    """Download required NLTK data"""
    logger.info("Setting up NLTK data...")
    
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download NLTK data: {str(e)}")
        return False

def run_data_processing():
    """Run data processing step"""
    return run_command(
        "python src/data_processing.py",
        "Data Processing"
    )

def run_model_training():
    """Run model training step"""
    return run_command(
        "python src/model_training.py",
        "Model Training"
    )

def test_prediction():
    """Test the prediction module"""
    logger.info("Testing prediction module...")
    
    try:
        # Add src to path
        sys.path.append('src')
        from prediction import create_sample_predictions
        
        create_sample_predictions()
        logger.info("✅ Prediction module test completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Prediction module test failed: {str(e)}")
        return False

def create_startup_script():
    """Create a script to start the API"""
    startup_script = """#!/bin/bash
# start_api.py
import uvicorn
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

if __name__ == "__main__":
    # Start the API server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8005,
        reload=True,
        log_level="info"
    )
"""
    try:
        with open('start_api.py', 'w') as f:
            f.write(startup_script)
        logger.info("✅ API startup script created: start_api.py")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create startup script: {str(e)}")
        return False

def main():
    """Run the complete pipeline"""
    logger.info("="*60)
    logger.info("STARTING RESTAURANT RATING PREDICTOR MLOPS PIPELINE")
    logger.info("="*60)
    
    steps = [
        ("Prerequisites Check", check_prerequisites),
        ("NLTK Setup", setup_nltk),
        ("Data Processing", run_data_processing),
        ("Model Training", run_model_training),
        ("Prediction Test", test_prediction),
        ("API Script Creation", create_startup_script)
    ]
    
    failed_steps = []
    
    for step_name, step_function in steps:
        logger.info(f"\n{'-'*40}")
        logger.info(f"STEP: {step_name}")
        logger.info(f"{'-'*40}")
        
        if not step_function():
            failed_steps.append(step_name)
            logger.error(f"❌ Step '{step_name}' failed!")
            
            # Ask if user wants to continue
            response = input(f"\nStep '{step_name}' failed. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                logger.error("Pipeline execution stopped by user")
                return False
        else:
            logger.info(f"✅ Step '{step_name}' completed successfully")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*60)
    
    if failed_steps:
        logger.warning(f"Pipeline completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            logger.warning(f"  ❌ {step}")
    else:
        logger.info("🎉 All steps completed successfully!")
    
    logger.info("\nNext steps:")
    logger.info("1. Start the API server: python start_api.py")
    logger.info("2. Open http://localhost:8005/docs for API documentation")
    logger.info("3. Test the API with sample data from /sample_data endpoint")
    
    # Check if models directory has files
    if os.path.exists('models') and os.listdir('models'):
        logger.info("\n📁 Generated files:")
        for file in os.listdir('models'):
            logger.info(f"  - models/{file}")
    
    logger.info("\n🚀 Your MLOps pipeline is ready!")
    return len(failed_steps) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)