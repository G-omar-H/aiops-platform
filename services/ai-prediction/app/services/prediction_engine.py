"""
Core Prediction Engine for AIOps AI Prediction Service
Orchestrates ML models and provides intelligent predictions
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import numpy as np
import structlog

from app.config import Settings
from app.models import (
    MetricPoint, PredictionRequest, PredictionResponse, PredictionResult,
    PredictionType, ModelType, SystemContext
)
from app.services.model_manager import ModelManager
from app.services.data_processor import DataProcessor
from app.utils.feature_engineering import FeatureEngineer
from app.utils.prediction_postprocessor import PredictionPostprocessor


logger = structlog.get_logger()


class PredictionEngine:
    """
    Core prediction engine that orchestrates ML models for various prediction types
    """
    
    def __init__(self, settings: Settings, model_manager: ModelManager, data_processor: DataProcessor):
        self.settings = settings
        self.model_manager = model_manager
        self.data_processor = data_processor
        self.feature_engineer = FeatureEngineer(settings)
        self.postprocessor = PredictionPostprocessor(settings)
        
        # Prediction type to model mapping
        self.prediction_model_mapping = {
            PredictionType.FAILURE_PREDICTION: [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.RANDOM_FOREST],
            PredictionType.CAPACITY_PLANNING: [ModelType.PROPHET, ModelType.LSTM, ModelType.XGBOOST],
            PredictionType.PERFORMANCE_FORECAST: [ModelType.LSTM, ModelType.TRANSFORMER, ModelType.PROPHET],
            PredictionType.RESOURCE_OPTIMIZATION: [ModelType.XGBOOST, ModelType.RANDOM_FOREST, ModelType.SVM],
            PredictionType.ANOMALY_DETECTION: [ModelType.ISOLATION_FOREST, ModelType.AUTOENCODER, ModelType.SVM],
            PredictionType.TREND_ANALYSIS: [ModelType.PROPHET, ModelType.LSTM, ModelType.TRANSFORMER]
        }
        
        # Model performance cache
        self.model_performance_cache: Dict[str, Dict[str, float]] = {}
        self.prediction_cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        
        self._initialized = False
        self._healthy = False
    
    async def initialize(self):
        """Initialize the prediction engine"""
        try:
            logger.info("🧠 Initializing Prediction Engine...")
            
            # Initialize feature engineer
            await self.feature_engineer.initialize()
            
            # Initialize postprocessor
            await self.postprocessor.initialize()
            
            # Load model performance metrics
            await self._load_model_performance_metrics()
            
            # Warm up models
            await self._warmup_models()
            
            self._initialized = True
            self._healthy = True
            
            logger.info("✅ Prediction Engine initialized successfully")
            
        except Exception as e:
            logger.error("❌ Failed to initialize Prediction Engine", error=str(e))
            self._healthy = False
            raise
    
    async def predict(
        self,
        metrics: List[MetricPoint],
        system_id: str,
        prediction_type: PredictionType,
        time_horizon: int = 12,
        confidence_threshold: float = 0.85
    ) -> PredictionResponse:
        """
        Make predictions based on input metrics
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Making prediction",
                system_id=system_id,
                prediction_type=prediction_type.value,
                time_horizon=time_horizon,
                metrics_count=len(metrics)
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(system_id, prediction_type, metrics, time_horizon)
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                logger.info("Returning cached prediction", system_id=system_id)
                return cached_result
            
            # Preprocess and validate data
            processed_data = await self._preprocess_data(metrics, system_id)
            
            # Feature engineering
            features = await self.feature_engineer.engineer_features(
                processed_data, prediction_type, time_horizon
            )
            
            # Select best model for this prediction type and system
            selected_model = await self._select_optimal_model(
                prediction_type, system_id, features
            )
            
            # Make prediction using selected model
            raw_predictions = await self._make_model_prediction(
                selected_model, features, time_horizon
            )
            
            # Post-process predictions
            processed_predictions = await self.postprocessor.process_predictions(
                raw_predictions, prediction_type, confidence_threshold
            )
            
            # Create prediction results
            prediction_results = self._create_prediction_results(
                processed_predictions, time_horizon
            )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(prediction_results)
            
            # Create response
            response = PredictionResponse(
                system_id=system_id,
                prediction_type=prediction_type,
                model_used=selected_model.model_id,
                predictions=prediction_results,
                confidence=overall_confidence,
                metadata={
                    "model_type": selected_model.model_type.value,
                    "feature_count": len(features),
                    "time_horizon": time_horizon,
                    "preprocessing_steps": features.get("preprocessing_steps", []),
                    "model_version": selected_model.version
                },
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Cache the result
            self._cache_prediction(cache_key, response)
            
            logger.info(
                "Prediction completed successfully",
                system_id=system_id,
                confidence=overall_confidence,
                processing_time_ms=response.processing_time_ms
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "Prediction failed",
                system_id=system_id,
                prediction_type=prediction_type.value,
                error=str(e)
            )
            raise
    
    async def batch_predict(
        self,
        requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """
        Handle batch predictions with parallel processing
        """
        logger.info("Processing batch predictions", batch_size=len(requests))
        
        # Group requests by prediction type for efficiency
        grouped_requests = self._group_requests_by_type(requests)
        
        # Process each group in parallel
        tasks = []
        for prediction_type, type_requests in grouped_requests.items():
            task = self._process_prediction_group(prediction_type, type_requests)
            tasks.append(task)
        
        # Wait for all predictions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        all_responses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Batch prediction group failed", error=str(result))
                continue
            all_responses.extend(result)
        
        logger.info("Batch predictions completed", total_responses=len(all_responses))
        return all_responses
    
    async def _preprocess_data(self, metrics: List[MetricPoint], system_id: str) -> Dict[str, Any]:
        """Preprocess input metrics for prediction"""
        
        # Convert metrics to time series format
        time_series_data = await self.data_processor.metrics_to_timeseries(metrics)
        
        # Apply data quality checks
        quality_score = await self.data_processor.assess_data_quality(time_series_data)
        if quality_score < 0.7:
            logger.warning("Low data quality detected", quality_score=quality_score, system_id=system_id)
        
        # Handle missing values and outliers
        cleaned_data = await self.data_processor.clean_data(time_series_data)
        
        # Normalize and scale data
        normalized_data = await self.data_processor.normalize_data(cleaned_data)
        
        return {
            "raw_data": time_series_data,
            "cleaned_data": cleaned_data,
            "normalized_data": normalized_data,
            "quality_score": quality_score,
            "preprocessing_steps": ["cleaning", "normalization"]
        }
    
    async def _select_optimal_model(
        self,
        prediction_type: PredictionType,
        system_id: str,
        features: Dict[str, Any]
    ):
        """Select the best model for the given prediction task"""
        
        # Get candidate models for this prediction type
        candidate_models = self.prediction_model_mapping.get(prediction_type, [])
        
        if not candidate_models:
            raise ValueError(f"No models available for prediction type: {prediction_type}")
        
        # Check for system-specific models first
        system_model = await self.model_manager.get_system_specific_model(system_id, prediction_type)
        if system_model and system_model.status == "active":
            logger.info("Using system-specific model", system_id=system_id, model_id=system_model.model_id)
            return system_model
        
        # Select best general model based on performance metrics
        best_model = None
        best_score = 0.0
        
        for model_type in candidate_models:
            model = await self.model_manager.get_model(model_type, prediction_type)
            if not model or model.status != "active":
                continue
            
            # Calculate model score based on accuracy, performance, and feature compatibility
            score = await self._calculate_model_score(model, features, prediction_type)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        if not best_model:
            raise ValueError(f"No active models available for prediction type: {prediction_type}")
        
        logger.info(
            "Selected optimal model",
            model_id=best_model.model_id,
            model_type=best_model.model_type.value,
            score=best_score
        )
        
        return best_model
    
    async def _make_model_prediction(
        self,
        model,
        features: Dict[str, Any],
        time_horizon: int
    ) -> Dict[str, Any]:
        """Make prediction using the selected model"""
        
        # Prepare model input
        model_input = await self._prepare_model_input(model, features)
        
        # Make prediction
        prediction_result = await model.predict(model_input, time_horizon)
        
        return prediction_result
    
    async def _prepare_model_input(self, model, features: Dict[str, Any]) -> Any:
        """Prepare input data for the specific model type"""
        
        model_type = model.model_type
        
        if model_type in [ModelType.LSTM, ModelType.TRANSFORMER]:
            # For sequence models, prepare sequence data
            return await self._prepare_sequence_input(features)
        
        elif model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.SVM]:
            # For traditional ML models, prepare feature vectors
            return await self._prepare_feature_vector(features)
        
        elif model_type == ModelType.PROPHET:
            # For Prophet, prepare time series format
            return await self._prepare_prophet_input(features)
        
        elif model_type in [ModelType.ISOLATION_FOREST, ModelType.AUTOENCODER]:
            # For anomaly detection models
            return await self._prepare_anomaly_input(features)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    async def _prepare_sequence_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare sequence input for LSTM/Transformer models"""
        
        normalized_data = features["normalized_data"]
        sequence_length = self.settings.ml.default_sequence_length
        
        # Create sequences from the normalized data
        sequences = []
        for i in range(len(normalized_data) - sequence_length + 1):
            sequence = normalized_data[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    async def _prepare_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare feature vector for traditional ML models"""
        
        # Extract statistical features
        data = features["normalized_data"]
        
        feature_vector = [
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data),
            np.percentile(data, 25),
            np.percentile(data, 75),
            len(data)
        ]
        
        # Add engineered features if available
        if "engineered_features" in features:
            feature_vector.extend(features["engineered_features"])
        
        return np.array(feature_vector).reshape(1, -1)
    
    async def _prepare_prophet_input(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for Prophet model"""
        
        # Prophet expects ds (datestamp) and y (value) columns
        raw_data = features["raw_data"]
        
        return {
            "ds": [point["timestamp"] for point in raw_data],
            "y": [point["value"] for point in raw_data]
        }
    
    async def _prepare_anomaly_input(self, features: Dict[str, Any]) -> np.ndarray:
        """Prepare input for anomaly detection models"""
        
        normalized_data = features["normalized_data"]
        return np.array(normalized_data).reshape(-1, 1)
    
    def _create_prediction_results(
        self,
        processed_predictions: Dict[str, Any],
        time_horizon: int
    ) -> List[PredictionResult]:
        """Create prediction result objects"""
        
        results = []
        predictions = processed_predictions["predictions"]
        confidence_scores = processed_predictions.get("confidence_scores", [])
        confidence_intervals = processed_predictions.get("confidence_intervals", [])
        
        base_time = datetime.utcnow()
        
        for i in range(min(len(predictions), time_horizon)):
            timestamp = base_time + timedelta(hours=i + 1)
            confidence = confidence_scores[i] if i < len(confidence_scores) else 0.8
            
            result = PredictionResult(
                predicted_value=float(predictions[i]),
                confidence=float(confidence),
                timestamp=timestamp,
                lower_bound=confidence_intervals[i][0] if i < len(confidence_intervals) else None,
                upper_bound=confidence_intervals[i][1] if i < len(confidence_intervals) else None
            )
            results.append(result)
        
        return results
    
    def _calculate_overall_confidence(self, prediction_results: List[PredictionResult]) -> float:
        """Calculate overall confidence score for the predictions"""
        
        if not prediction_results:
            return 0.0
        
        confidences = [result.confidence for result in prediction_results]
        
        # Use weighted average with more recent predictions having higher weight
        weights = [1.0 / (i + 1) for i in range(len(confidences))]
        weighted_confidence = np.average(confidences, weights=weights)
        
        return float(weighted_confidence)
    
    async def _calculate_model_score(
        self,
        model,
        features: Dict[str, Any],
        prediction_type: PredictionType
    ) -> float:
        """Calculate a score for model selection"""
        
        base_score = 0.0
        
        # Base accuracy score
        if model.accuracy:
            base_score += model.accuracy * 0.4
        
        # Performance metrics
        performance_metrics = model.performance_metrics
        if "mae" in performance_metrics:
            # Lower MAE is better, so invert it
            mae_score = max(0, 1.0 - performance_metrics["mae"])
            base_score += mae_score * 0.3
        
        # Model freshness (newer models score higher)
        if model.training_date:
            days_old = (datetime.utcnow() - model.training_date).days
            freshness_score = max(0, 1.0 - days_old / 30.0)  # Decay over 30 days
            base_score += freshness_score * 0.2
        
        # Feature compatibility
        data_quality = features.get("quality_score", 0.5)
        base_score += data_quality * 0.1
        
        return min(1.0, base_score)
    
    def _generate_cache_key(
        self,
        system_id: str,
        prediction_type: PredictionType,
        metrics: List[MetricPoint],
        time_horizon: int
    ) -> str:
        """Generate cache key for prediction"""
        
        # Create a hash based on input parameters
        metrics_hash = hash(tuple(
            (m.name, m.value, m.timestamp.isoformat()) for m in metrics[-10:]  # Last 10 metrics
        ))
        
        return f"{system_id}:{prediction_type.value}:{time_horizon}:{metrics_hash}"
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[PredictionResponse]:
        """Get cached prediction if available and not expired"""
        
        if cache_key not in self.prediction_cache:
            return None
        
        cached_item = self.prediction_cache[cache_key]
        if time.time() - cached_item["timestamp"] > self.cache_ttl:
            del self.prediction_cache[cache_key]
            return None
        
        return cached_item["prediction"]
    
    def _cache_prediction(self, cache_key: str, prediction: PredictionResponse):
        """Cache prediction result"""
        
        self.prediction_cache[cache_key] = {
            "prediction": prediction,
            "timestamp": time.time()
        }
        
        # Clean old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        
        current_time = time.time()
        expired_keys = [
            key for key, value in self.prediction_cache.items()
            if current_time - value["timestamp"] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.prediction_cache[key]
    
    def _group_requests_by_type(
        self,
        requests: List[PredictionRequest]
    ) -> Dict[PredictionType, List[PredictionRequest]]:
        """Group prediction requests by type for batch processing"""
        
        grouped = {}
        for request in requests:
            if request.prediction_type not in grouped:
                grouped[request.prediction_type] = []
            grouped[request.prediction_type].append(request)
        
        return grouped
    
    async def _process_prediction_group(
        self,
        prediction_type: PredictionType,
        requests: List[PredictionRequest]
    ) -> List[PredictionResponse]:
        """Process a group of predictions of the same type"""
        
        # Create tasks for parallel processing
        tasks = []
        for request in requests:
            task = self.predict(
                metrics=request.metrics,
                system_id=request.system_id,
                prediction_type=request.prediction_type,
                time_horizon=request.time_horizon,
                confidence_threshold=request.confidence_threshold
            )
            tasks.append(task)
        
        # Process with concurrency limit
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_predictions)
        
        async def bounded_predict(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_predict(task) for task in tasks]
        results = await asyncio.gather(*bounded_tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = [
            result for result in results
            if not isinstance(result, Exception)
        ]
        
        return successful_results
    
    async def _load_model_performance_metrics(self):
        """Load historical model performance metrics"""
        
        try:
            # Load performance metrics from storage or cache
            # This would typically come from a database or monitoring system
            logger.info("Loading model performance metrics...")
            
            # Placeholder implementation
            self.model_performance_cache = {
                "lstm_failure_prediction": {"accuracy": 0.85, "mae": 0.12},
                "transformer_performance_forecast": {"accuracy": 0.88, "mae": 0.09},
                "prophet_capacity_planning": {"accuracy": 0.82, "mae": 0.15}
            }
            
        except Exception as e:
            logger.warning("Failed to load model performance metrics", error=str(e))
    
    async def _warmup_models(self):
        """Warm up models for faster initial predictions"""
        
        try:
            logger.info("Warming up models...")
            
            # Get all active models and make dummy predictions
            active_models = await self.model_manager.get_active_models()
            
            for model in active_models[:3]:  # Warm up top 3 models
                try:
                    # Create dummy data for warmup
                    dummy_features = {"normalized_data": [0.5] * 100}
                    dummy_input = await self._prepare_model_input(model, dummy_features)
                    
                    # Make dummy prediction
                    await model.predict(dummy_input, 1)
                    
                    logger.debug("Model warmed up", model_id=model.model_id)
                    
                except Exception as e:
                    logger.warning("Failed to warm up model", model_id=model.model_id, error=str(e))
            
        except Exception as e:
            logger.warning("Model warmup failed", error=str(e))
    
    def is_healthy(self) -> bool:
        """Check if the prediction engine is healthy"""
        return self._healthy and self._initialized
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Prediction Engine...")
        
        # Clear caches
        self.prediction_cache.clear()
        self.model_performance_cache.clear()
        
        # Cleanup components
        if hasattr(self.feature_engineer, 'cleanup'):
            await self.feature_engineer.cleanup()
        
        if hasattr(self.postprocessor, 'cleanup'):
            await self.postprocessor.cleanup()
        
        self._healthy = False
        logger.info("Prediction Engine cleanup completed") 