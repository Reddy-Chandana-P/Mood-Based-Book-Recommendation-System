import os
import logging
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from recommender import recommend, detect_mood
from gemini_explainer import explain_recommendation

FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

app = FastAPI(title="Book Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    user_text: str
    top_n: int = 6

class ExplainRequest(BaseModel):
    book: dict
    user_text: str
    mood: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def get_recommendations(req: RecommendRequest):
    if not req.user_text.strip():
        raise HTTPException(status_code=400, detail="user_text cannot be empty")
    try:
        books = recommend(req.user_text, req.top_n)
        return {"mood": books[0]["mood"] if books else "unknown", "books": books}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def get_explanation(req: ExplainRequest):
    try:
        explanation = explain_recommendation(req.book, req.user_text, req.mood)
        return {"explanation": explanation}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/moods")
def get_moods():
    from data_loader import MOOD_GENRE_MAP
    return {"moods": list(MOOD_GENRE_MAP.keys())}

@app.get("/")
def serve_frontend():
    index = os.path.join(FRONTEND_PATH, "index.html")
    if not os.path.exists(index):
        return {"error": f"index.html not found at {index}"}
    return FileResponse(index)

if __name__ == "__main__":
    import uvicorn
    print("\n📚 Book Recommender API")
    print("   Running at http://localhost:8000\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
