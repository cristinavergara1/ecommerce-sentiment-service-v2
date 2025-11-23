from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes.sentiment import router as sentiment_router
from .core.logging import configure_logging

app = FastAPI(
    title="Sentiment Analysis Microservice",
    description="API para análisis de sentimientos usando XLMRoBERTa",
    version="1.0.0"
)

# Configuración de CORS: permite todas las fuentes (ajusta según tus necesidades)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes poner aquí los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

configure_logging()

app.include_router(sentiment_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis Microservice"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "sentiment-analysis-microservice"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)