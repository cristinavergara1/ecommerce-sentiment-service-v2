from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import logging
import numpy as np
from scipy.special import softmax
import torch
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SentimentService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.load_model()

    def preprocess(self, text):
        """Preprocess text (username and link placeholders)"""
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def load_model(self):
        """Load the sentiment analysis model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.config = AutoConfig.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of the given text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment analysis results
        """
        try:
            # Preprocess the text
            processed_text = self.preprocess(text)
            
            # Tokenize
            encoded_input = self.tokenizer(processed_text, return_tensors='pt')
            
            # Get model output
            with torch.no_grad():
                output = self.model(**encoded_input)
            
            # Get scores
            scores = output.logits[0].detach().numpy()
            scores = softmax(scores)
            
            # Create ranking
            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            
            # Format results
            results = []
            for i in range(scores.shape[0]):
                label = self.config.id2label[ranking[i]]
                score = float(scores[ranking[i]])
                results.append({
                    "label": label,
                    "score": round(score, 4)
                })
            
            return {
                "text": text,
                "processed_text": processed_text,
                "results": results,
                "top_sentiment": results[0]["label"],
                "confidence": results[0]["score"]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise

# Instancia global del servicio
_sentiment_service = None

def get_sentiment_service() -> SentimentService:
    global _sentiment_service
    if _sentiment_service is None:
        _sentiment_service = SentimentService()
    return _sentiment_service

def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Función para analizar el sentimiento de un texto.
    
    Args:
        text (str): Texto a analizar
        
    Returns:
        Dict[str, Any]: Resultados formateados con sentimiento predicho y puntuaciones
    """
    service = get_sentiment_service()
    return service.analyze_sentiment(text)