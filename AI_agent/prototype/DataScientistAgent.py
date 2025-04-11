import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    path: str
    type: str  # csv, json, excel, etc.
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    description: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class ModelInfo:
    """Information about a trained model."""
    name: str
    type: str
    features: List[str]
    target: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    created_at: datetime
    path: Optional[str] = None
    description: Optional[str] = None

class DataScientistAgent:
    """Agent specialized in data science tasks including analysis, visualization, and modeling."""
    
    def __init__(self,
                 workspace_dir: str = "workspace",
                 log_dir: str = "logs",
                 log_level: int = logging.INFO):
        """
        Initialize the data scientist agent.
        
        Args:
            workspace_dir: Directory for saving models and artifacts
            log_dir: Directory for log files
            log_level: Logging level
        """
        # Set up logging
        self._setup_logging(log_dir, log_level)
        
        self.logger.info("Initializing DataScientistAgent")
        
        # Set up workspace
        self.workspace_dir = Path(workspace_dir)
        self.models_dir = self.workspace_dir / "models"
        self.plots_dir = self.workspace_dir / "plots"
        self.data_dir = self.workspace_dir / "data"
        
        # Create directories
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Workspace initialized at: {self.workspace_dir}")
        
        # Track loaded datasets and models
        self.datasets: Dict[str, DatasetInfo] = {}
        self.models: Dict[str, ModelInfo] = {}
        
        self.logger.info("DataScientistAgent initialization completed")

    def _setup_logging(self, log_dir: str, log_level: int):
        """Set up logging configuration."""
        # Create logger
        self.logger = logging.getLogger("DataScientistAgent")
        self.logger.setLevel(log_level)
        
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # File handler for operations
        ops_handler = logging.FileHandler(
            log_path / "data_scientist_operations.log"
        )
        ops_handler.setLevel(log_level)
        ops_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(ops_handler)
        
        # File handler for errors
        error_handler = logging.FileHandler(
            log_path / "data_scientist_errors.log"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s'
        ))
        self.logger.addHandler(error_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
        
        self.logger.info("Logging setup completed")

    def load_dataset(self, path: str, name: Optional[str] = None) -> DatasetInfo:
        """
        Load a dataset from file and analyze its properties.
        
        Args:
            path: Path to the dataset file
            name: Optional name for the dataset
            
        Returns:
            DatasetInfo object containing dataset information
        """
        self.logger.info(f"Loading dataset from: {path}")
        try:
            file_path = Path(path)
            if not name:
                name = file_path.stem
                
            # Load dataset based on file type
            file_type = file_path.suffix.lower()
            if file_type == '.csv':
                df = pd.read_csv(file_path)
            elif file_type == '.json':
                df = pd.read_json(file_path)
            elif file_type in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            self.logger.debug(f"Dataset loaded successfully: {df.shape}")
            
            # Analyze dataset
            dataset_info = DatasetInfo(
                name=name,
                path=str(file_path),
                type=file_type[1:],  # Remove leading dot
                shape=df.shape,
                columns=list(df.columns),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                missing_values={col: df[col].isna().sum() for col in df.columns}
            )
            
            # Store dataset info
            self.datasets[name] = dataset_info
            
            # Save a copy in workspace
            workspace_path = self.data_dir / f"{name}{file_type}"
            df.to_csv(workspace_path, index=False)
            self.logger.info(f"Dataset copied to workspace: {workspace_path}")
            
            return dataset_info
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}", exc_info=True)
            raise

    def analyze_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Perform exploratory data analysis on a dataset.
        
        Args:
            dataset_name: Name of the dataset to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info(f"Analyzing dataset: {dataset_name}")
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset not found: {dataset_name}")
                
            dataset_info = self.datasets[dataset_name]
            df = pd.read_csv(self.data_dir / f"{dataset_name}.csv")
            
            analysis = {
                "basic_stats": {},
                "correlations": {},
                "distributions": {},
                "unique_values": {}
            }
            
            # Basic statistics
            self.logger.debug("Computing basic statistics")
            analysis["basic_stats"] = df.describe().to_dict()
            
            # Correlations for numeric columns
            self.logger.debug("Computing correlations")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["correlations"] = df[numeric_cols].corr().to_dict()
                
                # Create correlation heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
                plt.title(f"Correlation Heatmap - {dataset_name}")
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"{dataset_name}_correlations.png")
                plt.close()
            
            # Distribution plots for numeric columns
            self.logger.debug("Creating distribution plots")
            for col in numeric_cols:
                plt.figure(figsize=(8, 6))
                sns.histplot(df[col], kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"{dataset_name}_{col}_dist.png")
                plt.close()
                
                analysis["distributions"][col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "skew": df[col].skew()
                }
            
            # Unique values for categorical columns
            self.logger.debug("Analyzing categorical columns")
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                analysis["unique_values"][col] = df[col].value_counts().to_dict()
                
                # Create bar plots for categorical columns
                plt.figure(figsize=(10, 6))
                df[col].value_counts().plot(kind='bar')
                plt.title(f"Value Counts - {col}")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f"{dataset_name}_{col}_counts.png")
                plt.close()
            
            self.logger.info("Dataset analysis completed successfully")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing dataset: {str(e)}", exc_info=True)
            raise

    def train_model(self,
                   dataset_name: str,
                   model_type: str,
                   target_column: str,
                   feature_columns: List[str],
                   model_params: Dict[str, Any] = None,
                   test_size: float = 0.2) -> ModelInfo:
        """
        Train a machine learning model on the specified dataset.
        
        Args:
            dataset_name: Name of the dataset to use
            model_type: Type of model to train (e.g., 'linear', 'random_forest', etc.)
            target_column: Name of the target column
            feature_columns: List of feature column names
            model_params: Optional model parameters
            test_size: Proportion of data to use for testing
            
        Returns:
            ModelInfo object containing model information
        """
        self.logger.info(f"Training {model_type} model on dataset: {dataset_name}")
        try:
            if dataset_name not in self.datasets:
                raise ValueError(f"Dataset not found: {dataset_name}")
            
            # Load data
            df = pd.read_csv(self.data_dir / f"{dataset_name}.csv")
            
            # Prepare features and target
            X = df[feature_columns]
            y = df[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize model
            if model_type.lower() == 'linear':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression(**(model_params or {}))
            elif model_type.lower() == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(**(model_params or {}))
            elif model_type.lower() == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBRegressor(**(model_params or {}))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            self.logger.debug("Training model")
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Create model info
            model_name = f"{dataset_name}_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_info = ModelInfo(
                name=model_name,
                type=model_type,
                features=feature_columns,
                target=target_column,
                metrics=metrics,
                parameters=model_params or {},
                created_at=datetime.now()
            )
            
            # Save model and artifacts
            model_dir = self.models_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            dump(model, model_dir / "model.joblib")
            dump(scaler, model_dir / "scaler.joblib")
            
            with open(model_dir / "info.json", "w") as f:
                json.dump(model_info.__dict__, f, default=str)
            
            # Store model info
            self.models[model_name] = model_info
            
            # Create performance plots
            if model_type.lower() in ['linear', 'random_forest', 'xgboost']:
                plt.figure(figsize=(8, 6))
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{model_type} - Actual vs Predicted")
                plt.tight_layout()
                plt.savefig(model_dir / "actual_vs_predicted.png")
                plt.close()
            
            self.logger.info(f"Model training completed. Metrics: {metrics}")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}", exc_info=True)
            raise

    def predict(self,
               model_name: str,
               data: Union[pd.DataFrame, Dict[str, Any]]) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            data: Input data for prediction
            
        Returns:
            Array of predictions
        """
        self.logger.info(f"Making predictions using model: {model_name}")
        try:
            if model_name not in self.models:
                raise ValueError(f"Model not found: {model_name}")
            
            model_info = self.models[model_name]
            model_dir = self.models_dir / model_name
            
            # Load model and scaler
            model = load(model_dir / "model.joblib")
            scaler = load(model_dir / "scaler.joblib")
            
            # Prepare input data
            if isinstance(data, dict):
                data = pd.DataFrame([data])
            
            if not all(col in data.columns for col in model_info.features):
                missing = set(model_info.features) - set(data.columns)
                raise ValueError(f"Missing features in input data: {missing}")
            
            # Scale features
            X = scaler.transform(data[model_info.features])
            
            # Make predictions
            predictions = model.predict(X)
            
            self.logger.info("Predictions generated successfully")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            raise

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a specific dataset."""
        return self.datasets.get(dataset_name)

    def list_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())

    def list_datasets(self) -> List[str]:
        """Get list of available datasets."""
        return list(self.datasets.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent's state."""
        self.logger.info("Gathering agent statistics")
        try:
            stats = {
                "datasets": {
                    "count": len(self.datasets),
                    "types": {},
                    "total_rows": 0,
                    "total_columns": 0
                },
                "models": {
                    "count": len(self.models),
                    "types": {},
                    "average_metrics": {}
                }
            }
            
            # Dataset statistics
            for dataset in self.datasets.values():
                stats["datasets"]["types"][dataset.type] = \
                    stats["datasets"]["types"].get(dataset.type, 0) + 1
                stats["datasets"]["total_rows"] += dataset.shape[0]
                stats["datasets"]["total_columns"] += dataset.shape[1]
            
            # Model statistics
            for model in self.models.values():
                stats["models"]["types"][model.type] = \
                    stats["models"]["types"].get(model.type, 0) + 1
                
                # Aggregate metrics
                for metric, value in model.metrics.items():
                    if metric not in stats["models"]["average_metrics"]:
                        stats["models"]["average_metrics"][metric] = []
                    stats["models"]["average_metrics"][metric].append(value)
            
            # Calculate average metrics
            for metric, values in stats["models"]["average_metrics"].items():
                stats["models"]["average_metrics"][metric] = np.mean(values)
            
            self.logger.info("Statistics gathered successfully")
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {str(e)}", exc_info=True)
            return {}
