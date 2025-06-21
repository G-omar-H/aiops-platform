"""
Prediction API Routes

Handles real-time and batch predictions, anomaly detection, and model management.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import get_settings
from app.models.prediction import (
    PredictionRequest, 
    PredictionResponse, 
    BatchPredictionRequest,
    BatchPredictionResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse
)
from app.services.ml_service import MLService
from app.services.auth_service import verify_token

logger = logging.getLogger(__name__)
settings = get_settings()
security = HTTPBearer()

router = APIRouter()


def get_ml_service():
    """Dependency to get ML service instance."""
    from main import app
    return app.state.ml_service


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and get current user."""
    return await verify_token(credentials.credentials)


@router.post("/predict", response_model=PredictionResponse)
async def predict_failure(
    request: PredictionRequest,
    ml_service: MLService = Depends(get_ml_service),
    user: dict = Depends(get_current_user)
):
    """
    Predict system failure probability using ensemble ML models.
    
    **Real-time prediction endpoint for system failure analysis.**
    
    - **Analyzes current system metrics**
    - **Returns probability, confidence, and recommendations**
    - **Supports multiple ML algorithms (LSTM, Transformer, Random Forest)**
    - **Provides actionable insights and contributing factors**
    """
    try:
        logger.info(f"Prediction request from user {user.get('user_id', 'unknown')}")
        
        # Validate request
        if not request.metrics:
            raise HTTPException(status_code=400, detail="Metrics data is required")
        
        # Make prediction
        prediction = await ml_service.predict_failure(request)
        
        logger.info(f"Prediction completed: {prediction.failure_probability:.3f} probability")
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLService = Depends(get_ml_service),
    user: dict = Depends(get_current_user)
):
    """
    Perform batch predictions for multiple system instances.
    
    **Batch processing endpoint for large-scale prediction analysis.**
    
    - **Processes multiple prediction requests efficiently**
    - **Supports asynchronous processing for large batches**
    - **Returns aggregated insights and statistics**
    - **Ideal for scheduled health checks and bulk analysis**
    """
    try:
        logger.info(f"Batch prediction request: {len(request.requests)} items")
        
        if len(request.requests) > settings.ML_PREDICTION_BATCH_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size exceeds limit of {settings.ML_PREDICTION_BATCH_SIZE}"
            )
        
        # Process batch predictions
        predictions = []
        failed_predictions = []
        
        for i, pred_request in enumerate(request.requests):
            try:
                prediction = await ml_service.predict_failure(pred_request)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Individual prediction {i} failed: {e}")
                failed_predictions.append({
                    'index': i,
                    'error': str(e),
                    'request_id': pred_request.system_id
                })
        
        # Calculate batch statistics
        if predictions:
            probabilities = [p.failure_probability for p in predictions]
            avg_probability = sum(probabilities) / len(probabilities)
            max_probability = max(probabilities)
            high_risk_count = sum(1 for p in probabilities if p > 0.7)
        else:
            avg_probability = 0.0
            max_probability = 0.0
            high_risk_count = 0
        
        response = BatchPredictionResponse(
            batch_id=f"batch_{datetime.utcnow().timestamp()}",
            predictions=predictions,
            failed_predictions=failed_predictions,
            summary={
                'total_requests': len(request.requests),
                'successful_predictions': len(predictions),
                'failed_predictions': len(failed_predictions),
                'average_failure_probability': avg_probability,
                'max_failure_probability': max_probability,
                'high_risk_systems': high_risk_count
            }
        )
        
        logger.info(f"Batch prediction completed: {len(predictions)}/{len(request.requests)} successful")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@router.post("/anomaly/detect", response_model=AnomalyDetectionResponse)
async def detect_anomaly(
    request: AnomalyDetectionRequest,
    ml_service: MLService = Depends(get_ml_service),
    user: dict = Depends(get_current_user)
):
    """
    Detect anomalies in system metrics using machine learning.
    
    **Real-time anomaly detection for system health monitoring.**
    
    - **Uses Isolation Forest and statistical methods**
    - **Identifies unusual patterns in metrics**
    - **Provides severity scoring and alerts**
    - **Complements failure prediction with anomaly insights**
    """
    try:
        logger.info(f"Anomaly detection request from user {user.get('user_id', 'unknown')}")
        
        # Validate request
        if not request.metrics:
            raise HTTPException(status_code=400, detail="Metrics data is required")
        
        # Detect anomaly
        anomaly_result = await ml_service.detect_anomaly(request.metrics)
        
        response = AnomalyDetectionResponse(
            detection_id=f"anomaly_{datetime.utcnow().timestamp()}",
            anomaly_detected=anomaly_result.get('anomaly_detected', False),
            anomaly_score=anomaly_result.get('anomaly_score', 0.0),
            severity=anomaly_result.get('severity', 'low'),
            affected_metrics=request.metrics,
            timestamp=datetime.utcnow()
        )
        
        logger.info(f"Anomaly detection completed: {response.anomaly_detected}")
        return response
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.get("/models/performance")
async def get_model_performance(
    ml_service: MLService = Depends(get_ml_service),
    user: dict = Depends(get_current_user)
):
    """
    Get performance metrics for all ML models.
    
    **Retrieve current model accuracy, precision, recall, and F1 scores.**
    
    - **Real-time model performance monitoring**
    - **Helps identify when models need retraining**
    - **Supports ML ops and model lifecycle management**
    """
    try:
        metrics = await ml_service.get_model_performance()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'models': {
                name: {
                    'accuracy': model_metrics.accuracy,
                    'precision': model_metrics.precision,
                    'recall': model_metrics.recall,
                    'f1_score': model_metrics.f1_score,
                    'last_updated': model_metrics.last_updated.isoformat()
                }
                for name, model_metrics in metrics.items()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model performance: {str(e)}")


@router.post("/models/retrain")
async def retrain_models(
    background_tasks: BackgroundTasks,
    ml_service: MLService = Depends(get_ml_service),
    user: dict = Depends(get_current_user),
    model_names: Optional[List[str]] = Query(default=None, description="Specific models to retrain")
):
    """
    Trigger model retraining with latest data.
    
    **Initiate ML model retraining for improved accuracy.**
    
    - **Supports selective model retraining**
    - **Asynchronous processing for large training jobs**
    - **Updates model performance metrics**
    - **Maintains model versioning and rollback capability**
    """
    try:
        # Check permissions (only admin users can retrain models)
        if user.get('role') != 'admin':
            raise HTTPException(status_code=403, detail="Insufficient permissions for model retraining")
        
        logger.info(f"Model retraining requested by user {user.get('user_id')}")
        
        # Start retraining in background
        task_id = f"retrain_{datetime.utcnow().timestamp()}"
        
        # For now, we'll use synthetic data for retraining
        # In production, this would fetch real training data from the data pipeline
        background_tasks.add_task(
            _retrain_models_background,
            ml_service,
            task_id,
            model_names
        )
        
        return {
            'task_id': task_id,
            'status': 'initiated',
            'message': 'Model retraining started in background',
            'models': model_names or 'all',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model retraining failed: {str(e)}")


@router.get("/predictions/history")
async def get_prediction_history(
    system_id: Optional[str] = Query(default=None, description="Filter by system ID"),
    limit: int = Query(default=100, le=1000, description="Number of records to return"),
    user: dict = Depends(get_current_user)
):
    """
    Retrieve historical prediction data for analysis.
    
    **Access historical predictions for trend analysis and model validation.**
    
    - **Supports filtering by system ID and time range**
    - **Provides pagination for large datasets**
    - **Useful for performance analysis and reporting**
    """
    try:
        # This would typically query a database
        # For now, return a placeholder response
        
        return {
            'predictions': [],
            'total_count': 0,
            'system_id': system_id,
            'limit': limit,
            'timestamp': datetime.utcnow().isoformat(),
            'message': 'Historical prediction data would be returned here'
        }
        
    except Exception as e:
        logger.error(f"Failed to get prediction history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")


async def _retrain_models_background(ml_service: MLService, task_id: str, model_names: Optional[List[str]]):
    """Background task for model retraining."""
    try:
        logger.info(f"Starting background retraining task: {task_id}")
        
        # Generate synthetic training data
        # In production, this would fetch real data from your data pipeline
        import pandas as pd
        synthetic_data = ml_service._generate_synthetic_data(n_samples=5000)
        
        # Retrain models
        results = await ml_service.retrain_models(synthetic_data)
        
        logger.info(f"Retraining task {task_id} completed: {results}")
        
        # In production, you would store the results in a database
        # and potentially send notifications about completion
        
    except Exception as e:
        logger.error(f"Background retraining task {task_id} failed: {e}") 