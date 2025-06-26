# src/model_training.py
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, experiment_name="restaurant_rating_prediction"):
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Load processed data"""
        logger.info("Loading processed data...")
        X = np.load('data/processed/features.npy')
        y = np.load('data/processed/targets.npy')
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        logger.info("Splitting data...")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        logger.info("Training Linear Regression...")
        
        with mlflow.start_run(run_name="linear_regression"):
            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Store model
            self.models['linear_regression'] = {
                'model': model,
                'metrics': {
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
            }
            
            logger.info(f"Linear Regression - Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
            
            return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model with hyperparameter tuning"""
        logger.info("Training Random Forest with hyperparameter tuning...")
        
        with mlflow.start_run(run_name="random_forest"):
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            # Create base model
            rf = RandomForestRegressor(random_state=42)
            
            # Perform grid search
            logger.info("Performing grid search...")
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Log parameters and metrics
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("train_rmse", train_rmse)
            mlflow.log_metric("test_rmse", test_rmse)
            mlflow.log_metric("train_mae", train_mae)
            mlflow.log_metric("test_mae", test_mae)
            mlflow.log_metric("train_r2", train_r2)
            mlflow.log_metric("test_r2", test_r2)
            mlflow.log_metric("cv_score", -grid_search.best_score_)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Store model
            self.models['random_forest'] = {
                'model': best_model,
                'metrics': {
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                },
                'best_params': grid_search.best_params_
            }
            
            logger.info(f"Random Forest - Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            return best_model
    
    def select_best_model(self):
        """Select the best model based on test RMSE"""
        logger.info("Selecting best model...")
        
        best_rmse = float('inf')
        best_model_name = None
        
        for model_name, model_info in self.models.items():
            rmse = model_info['metrics']['test_rmse']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        return self.best_model, best_model_name
    
    def save_best_model(self):
        """Save the best model"""
        if self.best_model is None:
            logger.error("No best model selected!")
            return
        
        logger.info(f"Saving best model: {self.best_model_name}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        joblib.dump(self.best_model, 'models/best_model.pkl')
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'metrics': self.models[self.best_model_name]['metrics']
        }
        
        if 'best_params' in self.models[self.best_model_name]:
            model_info['best_params'] = self.models[self.best_model_name]['best_params']
        
        pd.Series(model_info).to_json('models/model_info.json')
        
        logger.info("Model saved successfully!")
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        evaluation_results = {
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2,
            'MAPE': mape
        }
        
        return evaluation_results
    
    def cross_validate_best_model(self, X, y, cv=5):
        """Perform cross-validation on the best model"""
        if self.best_model is None:
            logger.error("No best model selected!")
            return
        
        logger.info("Performing cross-validation...")
        
        cv_scores = cross_val_score(
            self.best_model, X, y, 
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        cv_rmse_scores = np.sqrt(-cv_scores)
        
        logger.info(f"Cross-validation RMSE: {cv_rmse_scores.mean():.4f} (+/- {cv_rmse_scores.std() * 2:.4f})")
        
        return cv_rmse_scores
    
    def train_all_models(self):
        """Train all models and select the best one"""
        logger.info("Starting model training pipeline...")
        
        # Load data
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        logger.info(f"Training set size: {X_train.shape[0]}")
        logger.info(f"Test set size: {X_test.shape[0]}")
        
        # Train models
        self.train_linear_regression(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        
        # Select best model
        best_model, best_model_name = self.select_best_model()
        
        # Cross-validate best model
        self.cross_validate_best_model(X, y)
        
        # Evaluate best model
        evaluation_results = self.evaluate_model(best_model, X_test, y_test)
        
        logger.info("Final evaluation results:")
        for metric, value in evaluation_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save best model
        self.save_best_model()
        
        return best_model, best_model_name, evaluation_results

def main():
    """Main training function"""
    trainer = ModelTrainer()
    best_model, best_model_name, evaluation_results = trainer.train_all_models()
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best Model: {best_model_name}")
    print("\nFinal Metrics:")
    for metric, value in evaluation_results.items():
        print(f"  {metric}: {value:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()