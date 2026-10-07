from fastapi import FastAPI, Query, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi.responses import PlainTextResponse
import time
import logging
from model_loader import RecommenderModel

app = FastAPI()

REQUESTS = Counter('requests_total', 'Total number of requests')
ERRORS = Counter('error_total', 'Total errors', ['code'])
DURATION = Histogram('request_duration_seconds', 'Request duration')

try:
    model = RecommenderModel('model.pkl')
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/recommend")
async def recommend(user_id: str = Query(..., description="User ID"),
                    top_n: int = Query(3, ge=1, le=20, description="Number of recommendations")):
    REQUESTS.inc()
    start_time = time.time()
    try:
        if model is None:
            ERRORS.labels(code='503').inc()
            raise HTTPException(status_code=503, detail="Model not loaded")
        recs = model.recommend(user_id, top_n)
        if recs is None:
            ERRORS.labels(code='404').inc()
            raise HTTPException(status_code=404, detail="User not found")
        result = {
            "user_id": user_id,
            "recommendations": [{"item_id": item_id, "score": score} for item_id, score in recs]
        }
        DURATION.observe(time.time() - start_time)
        return result
    except HTTPException:
        raise
    except Exception as e:
        ERRORS.labels(code='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(generate_latest(REGISTRY))
