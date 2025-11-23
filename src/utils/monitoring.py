"""Monitoring utilities for sentiment analysis ML logging."""
import httpx
from typing import Dict, Any, Optional
from datetime import datetime

# Configuration
MONITORING_SERVICE_URL = "http://localhost:8003/api/v1"


def categorize_confidence(score: float) -> str:
    """
    Convert numeric confidence score to categorical confidence level.
    Uses same thresholds as v1 service for consistency.

    Args:
        score: Numeric confidence score (0.0 to 1.0)

    Returns:
        str: Confidence category ('high', 'medium', or 'low')
    """
    if score > 0.8:
        return "high"
    elif score > 0.6:
        return "medium"
    else:
        return "low"


async def log_sentiment_prediction(
    text: str,
    prediction_result: Dict[str, Any],
    latency_ms: float,
    product_id: Optional[str] = None,
    user_id: Optional[str] = None,
    rating: Optional[int] = None,
    error: Optional[str] = None,
):
    """
    Log sentiment prediction to monitoring service.

    Args:
        text: Input text that was analyzed
        prediction_result: Result dictionary from analyze_sentiment
        latency_ms: Time taken for prediction in milliseconds
        product_id: Optional product ID for tracking
        user_id: Optional user ID for tracking
        rating: Optional rating (1-5) if available
        error: Optional error message if prediction failed
    """
    try:
        # Extract scores from prediction result (v2 format)
        all_scores = {}
        for score_obj in prediction_result.get("results", []):
            label = score_obj.get("label")
            score = score_obj.get("score")
            if label and score is not None:
                all_scores[label] = score

        # Get confidence score and categorize it
        confidence_score = prediction_result.get("confidence", 0.0)
        confidence_category = categorize_confidence(confidence_score)

        payload = {
            "text": text,
            "text_length": len(text),
            "predicted_sentiment": prediction_result.get("top_sentiment"),
            "confidence": confidence_category,
            "primary_score": confidence_score,
            "all_scores": all_scores,
            "product_id": product_id,
            "user_id": user_id,
            "rating": rating,
            "latency_ms": latency_ms,
            "error": error,
        }

        # Send async request with timeout
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.post(
                f"{MONITORING_SERVICE_URL}/predictions/sentiment",
                json=payload
            )
            response.raise_for_status()

    except httpx.TimeoutException:
        # Silent fail on timeout - don't block main request
        print(f"Warning: Monitoring service timeout")
    except httpx.HTTPError as e:
        # Silent fail on HTTP errors - don't block main request
        print(f"Warning: Failed to log sentiment prediction to monitoring: {e}")
    except Exception as e:
        # Catch all other exceptions - don't block main request
        print(f"Warning: Unexpected error logging to monitoring: {e}")
