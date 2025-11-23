from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from ...services.sentiment_service import analyze_sentiment
from ...models.response import SentimentResponse, SentimentScore
from ...utils.monitoring import log_sentiment_prediction
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

class SentimentRequest(BaseModel):
    text: str

@router.post("/analyze", response_model=SentimentResponse)
async def analyze_text_sentiment(request: SentimentRequest):
    start_time = datetime.now()
    error = None
    result = None

    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

        logger.info(f"Analizando texto: {request.text[:100]}...")
        result = analyze_sentiment(request.text)
        logger.info(f"Resultados obtenidos: {result}")

        # Convertir results a objetos SentimentScore con el formato correcto
        sentiment_scores = [
            SentimentScore(
                label=score["label"],
                score=score["score"],
                confidence=f"{score['score']:.4f}"
            ) for score in result["results"]
        ]

        return SentimentResponse(
            text=result["text"],
            predicted_sentiment=result["top_sentiment"],
            confidence=f"{result['confidence']:.4f}",
            primary_score=result["confidence"],
            all_scores=sentiment_scores
        )

    except HTTPException:
        error = "Validation error"
        raise
    except Exception as e:
        error = str(e)
        logger.error(f"Error en análisis de sentimiento: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

    finally:
        # Log to monitoring service (non-blocking)
        if result:
            try:
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                await log_sentiment_prediction(
                    text=request.text,
                    prediction_result=result,
                    latency_ms=latency_ms,
                    error=error,
                )
            except Exception as log_error:
                # Silent fail - monitoring should never break the main request
                logger.debug(f"Failed to log to monitoring: {log_error}")

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "sentiment-analysis"}