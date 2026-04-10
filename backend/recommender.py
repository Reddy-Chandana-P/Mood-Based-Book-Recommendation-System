from __future__ import annotations
import os
# Suppress noisy warnings before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_books, build_tfidf, get_genre_matches, MOOD_GENRE_MAP

# Load once at startup
_df = None
_vectorizer = None
_tfidf_matrix = None
_mood_classifier = None

MOOD_LABELS = list(MOOD_GENRE_MAP.keys())

def _init():
    global _df, _vectorizer, _tfidf_matrix, _mood_classifier
    if _df is None:
        print("📂 Loading books dataset...")
        _df = load_books()
        _vectorizer, _tfidf_matrix = build_tfidf(_df)
        print(f"✅ Loaded {len(_df)} books")
    if _mood_classifier is None:
        print("🧠 Loading NLP mood classifier...")
        try:
            _mood_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            print("✅ NLP model ready")
        except Exception as e:
            print(f"⚠️  NLP model failed: {e}")
            # Fallback: simple keyword-based mood detection
            _mood_classifier = None

def detect_mood(user_text: str) -> str:
    """Use zero-shot NLP to classify user text into a mood, with keyword fallback."""
    _init()
    if _mood_classifier is not None:
        result = _mood_classifier(user_text, candidate_labels=MOOD_LABELS)
        return result["labels"][0]
    # Keyword fallback
    text = user_text.lower()
    keyword_map = {
        "happy": ["happy", "joy", "fun", "excited", "great"],
        "sad": ["sad", "depressed", "down", "unhappy", "cry"],
        "adventurous": ["adventure", "explore", "exciting", "thrill"],
        "romantic": ["love", "romantic", "relationship", "heart"],
        "curious": ["curious", "learn", "know", "discover", "science"],
        "scared": ["scared", "fear", "horror", "dark", "creepy"],
        "motivated": ["motivated", "inspire", "goal", "success", "achieve"],
        "relaxed": ["relax", "calm", "peaceful", "chill", "stress"],
        "bored": ["bored", "nothing", "idle"],
        "nostalgic": ["nostalgic", "memories", "past", "childhood"],
    }
    for mood, keywords in keyword_map.items():
        if any(k in text for k in keywords):
            return mood
    return "curious"


def recommend(user_text: str, top_n: int = 6) -> list[dict]:
    """
    1. Detect mood from user text
    2. Filter books by mood→genre
    3. Rank by TF-IDF similarity to user text + rating score
    4. Return top N books
    """
    _init()

    mood = detect_mood(user_text)
    genre_df = get_genre_matches(_df, mood)

    # Get indices of genre-filtered books in original df
    indices = genre_df.index.tolist()

    # TF-IDF similarity between user text and filtered books
    user_vec = _vectorizer.transform([user_text])
    subset_matrix = _tfidf_matrix[indices]
    similarities = cosine_similarity(user_vec, subset_matrix).flatten()

    # Normalize rating (0-5 scale) and combine with similarity
    ratings = genre_df["rating"].values
    max_rating = ratings.max() if ratings.max() > 0 else 1
    norm_ratings = ratings / max_rating

    # Weighted score: 70% similarity, 30% rating
    scores = 0.7 * similarities + 0.3 * norm_ratings

    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []

    for i in top_indices:
        row = genre_df.iloc[i]
        results.append({
            "title": str(row.get("title", "Unknown")),
            "author": str(row.get("author", "Unknown")),
            "genre": str(row.get("genre", "")),
            "description": str(row.get("description", ""))[:400],
            "rating": float(row.get("rating", 0)),
            "thumbnail": str(row.get("thumbnail", "")),
            "score": float(scores[i]),
            "mood": mood,
        })

    return results
