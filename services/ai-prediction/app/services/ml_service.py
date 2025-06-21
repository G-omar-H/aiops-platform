"""
Machine Learning Service - Core AI/ML Operations

Handles model loading, prediction, training, and lifecycle management.
Supports multiple ML algorithms for different use cases.
"""

import asyncio
import logging
import pickle
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

import joblib
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app.core.config import get_settings
from app.models.prediction import PredictionRequest, PredictionResponse, ModelMetrics
from app.services.metrics_service import MetricsService

logger = logging.getLogger(__name__)
settings = get_settings()


class LSTMModel(nn.Module):
    """LSTM Neural Network for Time Series Prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, output_size: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class TransformerModel(nn.Module):
    """Transformer model for advanced time series analysis."""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=512,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x[-1, :, :]  # Take last timestep
        return self.output_layer(x)


class MLService:
    """Enterprise ML Service for Predictive Analytics."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.models_path = Path(settings.ML_MODELS_PATH)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering settings
        self.feature_columns = [
            'cpu_usage', 'memory_usage', 'disk_usage', 'network_io',
            'request_count', 'response_time', 'error_rate',
            'gc_frequency', 'thread_count', 'connection_count'
        ]
        
        # Model types and their configurations
        self.model_configs = {
            'lstm': {
                'type': 'pytorch',
                'class': LSTMModel,
                'params': {'input_size': len(self.feature_columns), 'hidden_size': 128}
            },
            'transformer': {
                'type': 'pytorch', 
                'class': TransformerModel,
                'params': {'input_size': len(self.feature_columns), 'd_model': 128}
            },
            'random_forest': {
                'type': 'sklearn',
                'class': RandomForestClassifier,
                'params': {'n_estimators': settings.RF_N_ESTIMATORS, 'max_depth': settings.RF_MAX_DEPTH}
            },
            'isolation_forest': {
                'type': 'sklearn',
                'class': IsolationForest,
                'params': {'contamination': 0.1, 'random_state': 42}
            }
        }
        
    async def load_models(self) -> None:
        """Load all available models from disk."""
        logger.info("Loading ML models...")
        
        for model_name, config in self.model_configs.items():
            try:
                model_path = self.models_path / f"{model_name}.pkl"
                scaler_path = self.models_path / f"{model_name}_scaler.pkl"
                
                if model_path.exists():
                    if config['type'] == 'pytorch':
                        # Load PyTorch models
                        model = config['class'](**config['params'])
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        model.eval()
                        self.models[model_name] = model
                    else:
                        # Load scikit-learn models
                        self.models[model_name] = joblib.load(model_path)
                    
                    # Load scaler if exists
                    if scaler_path.exists():
                        self.scalers[model_name] = joblib.load(scaler_path)
                    
                    logger.info(f"✅ Loaded model: {model_name}")
                else:
                    # Create and train new model
                    logger.info(f"Model {model_name} not found, will train new one")
                    await self._train_new_model(model_name)
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model {model_name}: {e}")
    
    async def predict_failure(self, request: PredictionRequest) -> PredictionResponse:
        """Predict system failure probability using ensemble of models."""
        try:
            # Prepare features
            features = await self._prepare_features(request)
            
            # Get predictions from all models
            predictions = {}
            confidence_scores = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name in ['lstm', 'transformer']:
                        pred, conf = await self._predict_pytorch(model, features, model_name)
                    else:
                        pred, conf = await self._predict_sklearn(model, features, model_name)
                    
                    predictions[model_name] = pred
                    confidence_scores[model_name] = conf
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for model {model_name}: {e}")
            
            # Ensemble prediction (weighted average)
            if predictions:
                ensemble_prediction = await self._ensemble_predict(predictions, confidence_scores)
                
                return PredictionResponse(
                    prediction_id=f"pred_{datetime.utcnow().timestamp()}",
                    failure_probability=ensemble_prediction['probability'],
                    confidence_score=ensemble_prediction['confidence'],
                    risk_level=self._calculate_risk_level(ensemble_prediction['probability']),
                    predicted_failure_time=self._estimate_failure_time(ensemble_prediction['probability']),
                    contributing_factors=await self._identify_contributing_factors(features),
                    model_predictions=predictions,
                    recommendations=await self._generate_recommendations(ensemble_prediction)
                )
            else:
                raise ValueError("No models available for prediction")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def detect_anomaly(self, metrics_data: Dict[str, float]) -> Dict[str, Any]:
        """Detect anomalies in system metrics."""
        try:
            if 'isolation_forest' not in self.models:
                return {'anomaly_detected': False, 'reason': 'Anomaly detection model not available'}
            
            # Prepare features for anomaly detection
            features = np.array([[metrics_data.get(col, 0) for col in self.feature_columns]])
            
            # Scale features if scaler is available
            if 'isolation_forest' in self.scalers:
                features = self.scalers['isolation_forest'].transform(features)
            
            # Predict anomaly
            anomaly_score = self.models['isolation_forest'].decision_function(features)[0]
            is_anomaly = self.models['isolation_forest'].predict(features)[0] == -1
            
            return {
                'anomaly_detected': is_anomaly,
                'anomaly_score': float(anomaly_score),
                'threshold': 0.0,
                'severity': 'high' if anomaly_score < -0.5 else 'medium' if anomaly_score < -0.2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {'anomaly_detected': False, 'error': str(e)}
    
    async def retrain_models(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Retrain models with new data."""
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                logger.info(f"Retraining model: {model_name}")
                metrics = await self._train_model(model_name, training_data)
                results[model_name] = {
                    'status': 'success',
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Failed to retrain {model_name}: {e}")
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return results
    
    async def get_model_performance(self) -> Dict[str, ModelMetrics]:
        """Get performance metrics for all models."""
        return self.model_metrics
    
    async def _prepare_features(self, request: PredictionRequest) -> np.ndarray:
        """Prepare features from prediction request."""
        # Extract features from metrics
        features = []
        for col in self.feature_columns:
            value = request.metrics.get(col, 0.0)
            features.append(value)
        
        return np.array(features).reshape(1, -1)
    
    async def _predict_pytorch(self, model: nn.Module, features: np.ndarray, model_name: str) -> Tuple[float, float]:
        """Make prediction using PyTorch model."""
        with torch.no_grad():
            # Scale features if scaler available
            if model_name in self.scalers:
                features = self.scalers[model_name].transform(features)
            
            # Convert to tensor
            if model_name == 'lstm' or model_name == 'transformer':
                # For sequence models, we need to create a sequence
                # For now, repeat the current features to create a sequence
                seq_length = settings.LSTM_SEQUENCE_LENGTH
                features_seq = np.repeat(features, seq_length, axis=0).reshape(1, seq_length, -1)
                tensor = torch.FloatTensor(features_seq)
            else:
                tensor = torch.FloatTensor(features)
            
            # Get prediction
            output = model(tensor)
            prediction = torch.sigmoid(output).item()  # Convert to probability
            confidence = min(max(abs(output.item()), 0.5), 1.0)  # Simple confidence measure
            
            return prediction, confidence
    
    async def _predict_sklearn(self, model: Any, features: np.ndarray, model_name: str) -> Tuple[float, float]:
        """Make prediction using scikit-learn model."""
        # Scale features if scaler available
        if model_name in self.scalers:
            features = self.scalers[model_name].transform(features)
        
        # Get prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            prediction = proba[1] if len(proba) > 1 else proba[0]
            confidence = max(proba)
        else:
            prediction = model.predict(features)[0]
            confidence = 0.8  # Default confidence for models without probability
        
        return float(prediction), float(confidence)
    
    async def _ensemble_predict(self, predictions: Dict[str, float], confidences: Dict[str, float]) -> Dict[str, float]:
        """Combine predictions from multiple models using weighted ensemble."""
        # Weights based on model confidence and historical performance
        weights = {
            'lstm': 0.3,
            'transformer': 0.3,
            'random_forest': 0.25,
            'isolation_forest': 0.15
        }
        
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.1) * confidences.get(model_name, 0.5)
            weighted_prediction += prediction * weight
            weighted_confidence += confidences[model_name] * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_prediction /= total_weight
            weighted_confidence /= total_weight
        
        return {
            'probability': weighted_prediction,
            'confidence': weighted_confidence
        }
    
    def _calculate_risk_level(self, probability: float) -> str:
        """Calculate risk level based on failure probability."""
        if probability >= 0.8:
            return 'critical'
        elif probability >= 0.6:
            return 'high'
        elif probability >= 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_failure_time(self, probability: float) -> Optional[datetime]:
        """Estimate when failure might occur based on probability."""
        if probability < 0.3:
            return None
        
        # Simple estimation: higher probability = sooner failure
        hours_until_failure = max(1, int(24 * (1 - probability)))
        return datetime.utcnow() + timedelta(hours=hours_until_failure)
    
    async def _identify_contributing_factors(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Identify which metrics contribute most to failure prediction."""
        factors = []
        
        for i, col in enumerate(self.feature_columns):
            value = features[0][i]
            # Simple threshold-based analysis
            if col in ['cpu_usage', 'memory_usage', 'disk_usage'] and value > 0.8:
                factors.append({
                    'metric': col,
                    'value': value,
                    'severity': 'high' if value > 0.9 else 'medium',
                    'impact': 'High resource utilization detected'
                })
            elif col == 'error_rate' and value > 0.05:
                factors.append({
                    'metric': col,
                    'value': value,
                    'severity': 'high',
                    'impact': 'Elevated error rate detected'
                })
        
        return factors
    
    async def _generate_recommendations(self, prediction: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on prediction."""
        recommendations = []
        
        probability = prediction['probability']
        
        if probability > 0.7:
            recommendations.extend([
                "Immediate attention required - Scale up resources",
                "Check system logs for errors",
                "Consider enabling circuit breakers",
                "Prepare rollback plan"
            ])
        elif probability > 0.5:
            recommendations.extend([
                "Monitor system closely",
                "Consider preventive scaling",
                "Review recent deployments"
            ])
        else:
            recommendations.append("System appears stable")
        
        return recommendations
    
    async def _train_new_model(self, model_name: str) -> None:
        """Train a new model with synthetic or available data."""
        logger.info(f"Training new model: {model_name}")
        
        # For demo purposes, create synthetic training data
        synthetic_data = self._generate_synthetic_data()
        await self._train_model(model_name, synthetic_data)
    
    async def _train_model(self, model_name: str, training_data: pd.DataFrame) -> ModelMetrics:
        """Train a specific model with provided data."""
        config = self.model_configs[model_name]
        
        # Prepare data
        X = training_data[self.feature_columns].values
        y = training_data['failure'].values if 'failure' in training_data.columns else np.random.randint(0, 2, len(X))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        if config['type'] == 'pytorch':
            model = await self._train_pytorch_model(model_name, X_train_scaled, y_train, X_test_scaled, y_test)
        else:
            model = config['class'](**config['params'])
            model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        if config['type'] == 'pytorch':
            y_pred = await self._predict_pytorch_batch(model, X_test_scaled, model_name)
        else:
            y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, np.round(y_pred)),
            precision=precision_score(y_test, np.round(y_pred), average='weighted'),
            recall=recall_score(y_test, np.round(y_pred), average='weighted'),
            f1_score=f1_score(y_test, np.round(y_pred), average='weighted'),
            last_updated=datetime.utcnow()
        )
        
        # Save model and scaler
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.model_metrics[model_name] = metrics
        
        await self._save_model(model_name, model, scaler)
        
        return metrics
    
    async def _train_pytorch_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                                 X_test: np.ndarray, y_test: np.ndarray) -> nn.Module:
        """Train a PyTorch model."""
        config = self.model_configs[model_name]
        model = config['class'](**config['params'])
        
        # Convert to tensors
        if model_name in ['lstm', 'transformer']:
            # Create sequences for time series models
            seq_length = settings.LSTM_SEQUENCE_LENGTH
            X_train_seq = self._create_sequences(X_train, seq_length)
            X_test_seq = self._create_sequences(X_test, seq_length)
            
            X_train_tensor = torch.FloatTensor(X_train_seq)
            y_train_tensor = torch.FloatTensor(y_train[:len(X_train_seq)]).unsqueeze(1)
        else:
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        epochs = settings.LSTM_EPOCHS if model_name in ['lstm', 'transformer'] else 50
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        model.eval()
        return model
    
    async def _predict_pytorch_batch(self, model: nn.Module, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make batch predictions with PyTorch model."""
        with torch.no_grad():
            if model_name in ['lstm', 'transformer']:
                seq_length = settings.LSTM_SEQUENCE_LENGTH
                X_seq = self._create_sequences(X, seq_length)
                X_tensor = torch.FloatTensor(X_seq)
            else:
                X_tensor = torch.FloatTensor(X)
            
            outputs = model(X_tensor)
            predictions = torch.sigmoid(outputs).numpy().flatten()
            
            return predictions
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> np.ndarray:
        """Create sequences for time series models."""
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    async def _save_model(self, model_name: str, model: Any, scaler: Any) -> None:
        """Save model and scaler to disk."""
        try:
            model_path = self.models_path / f"{model_name}.pkl"
            scaler_path = self.models_path / f"{model_name}_scaler.pkl"
            
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), model_path)
            else:
                joblib.dump(model, model_path)
            
            joblib.dump(scaler, scaler_path)
            
            logger.info(f"✅ Saved model: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model {model_name}: {e}")
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic training data for initial model training."""
        np.random.seed(42)
        
        data = {}
        
        # Generate synthetic metrics
        for col in self.feature_columns:
            if 'usage' in col:
                # Resource usage metrics (0-1)
                data[col] = np.random.beta(2, 5, n_samples)  # Skewed towards lower values
            elif col == 'error_rate':
                # Error rate (0-0.2)
                data[col] = np.random.exponential(0.02, n_samples)
            elif col == 'response_time':
                # Response time (ms)
                data[col] = np.random.lognormal(4, 1, n_samples)
            else:
                # Other metrics
                data[col] = np.random.normal(50, 20, n_samples)
        
        # Generate failure labels based on thresholds
        df = pd.DataFrame(data)
        
        # Create failure conditions
        failure_conditions = (
            (df['cpu_usage'] > 0.8) |
            (df['memory_usage'] > 0.9) |
            (df['error_rate'] > 0.1) |
            (df['response_time'] > 2000)
        )
        
        df['failure'] = failure_conditions.astype(int)
        
        # Add some noise
        noise_indices = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
        df.loc[noise_indices, 'failure'] = 1 - df.loc[noise_indices, 'failure']
        
        return df
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ML service...")
        self.models.clear()
        self.scalers.clear()
        self.model_metrics.clear() 