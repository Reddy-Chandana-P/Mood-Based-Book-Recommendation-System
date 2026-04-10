from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_books, build_tfidf, get_genre_matches, MOOD_GENRE_MAP

_df         = None
_vectorizer = None
_tfidf_matrix = None
_bayes_C    = None
_bayes_m    = None

MOOD_KEYWORDS: dict[str, list[str]] = {
    "happy":       ["happy","joy","joyful","fun","excited","great","wonderful",
                    "cheerful","good","positive","laugh","smile","upbeat","light"],
    "sad":         ["sad","depressed","unhappy","cry","grief","lonely",
                    "heartbroken","down","miserable","upset","hopeless","lost","empty"],
    "adventurous": ["adventure","explore","exciting","thrill","journey","travel",
                    "quest","wild","daring","action","epic","bold","brave"],
    "romantic":    ["love","romantic","romance","relationship","heart","crush",
                    "date","passion","affection","soulmate","feelings","tender"],
    "curious":     ["curious","learn","discover","science","history","know",
                    "understand","facts","research","explore","wonder","question"],
    "scared":      ["scared","fear","horror","dark","creepy","terrified",
                    "nightmare","ghost","spooky","eerie","chilling","dread"],
    "motivated":   ["motivated","inspire","goal","success","achieve","hustle",
                    "ambition","driven","focus","grind","productive","grow"],
    "relaxed":     ["relax","calm","peaceful","chill","stress","tired",
                    "unwind","quiet","cozy","lazy","slow","breathe","rest"],
    "bored":       ["bored","nothing","idle","dull","monotonous","restless",
                    "stuck","uninspired","flat"],
    "nostalgic":   ["nostalgic","memories","past","childhood","remember",
                    "old","vintage","classic","miss"],
    "stressed":    ["stressed","anxious","overwhelmed","pressure","tense",
                    "worried","nervous","burnout","exhausted","panic"],
    "inspired":    ["inspired","creative","moved","uplifted","enlightened",
                    "awakened","purpose","meaning"],
}


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return default


def _safe_int(v: object) -> int:
    try:
        f = float(v)
        return 0 if np.isnan(f) else int(f)
    except Exception:
        return 0


def _init() -> None:
    global _df, _vectorizer, _tfidf_matrix, _bayes_C, _bayes_m
    if _df is not None:
        return
    _df = load_books()
    print("🔧 Building TF-IDF index...")
    _vectorizer, _tfidf_matrix = build_tfidf(_df)
    # Precompute Bayesian constants once
    _bayes_C = float(_df["ratings_count"].median())
    _bayes_m = float(_df["rating"].mean())
    print("✅ Ready to recommend")


def detect_mood(user_text: str) -> str:
    text = user_text.lower()
    scores = {
        mood: sum(1 for k in kws if k in text)
        for mood, kws in MOOD_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "curious"


def recommend(user_text: str, top_n: int = 6) -> list[dict]:
    _init()

    mood     = detect_mood(user_text)
    genre_df = get_genre_matches(_df, mood)

    # Guard: if genre filter returns nothing fall back to full df
    if genre_df.empty:
        genre_df = _df

    indices       = genre_df.index.tolist()
    user_vec      = _vectorizer.transform([user_text])
    subset_matrix = _tfidf_matrix[indices]
    similarities  = cosine_similarity(user_vec, subset_matrix).flatten()

    # Bayesian average (IMDb-style) — rewards popular high-rated books
    genre_df = genre_df.copy()
    rc = genre_df["ratings_count"].fillna(0).values.astype(float)
    rt = genre_df["rating"].fillna(0).values.astype(float)
    bayesian = (rc * rt + _bayes_C * _bayes_m) / (rc + _bayes_C + 1e-9)
    max_b    = bayesian.max()
    norm_b   = bayesian / (max_b if max_b > 0 else 1.0)

    # 50% content similarity + 50% bayesian quality
    scores      = 0.50 * similarities + 0.50 * norm_b
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []
    for i in top_indices:
        row = genre_df.iloc[i]

        # Clean genre: take first 3 space-separated tags, humanise
        genre_raw     = str(row.get("genre", ""))
        genre_display = ", ".join(
            t.strip().replace("-", " ").title()
            for t in genre_raw.split()[:3]
            if t.strip()
        ) or "General"

        # Safe thumbnail — skip obviously broken values
        thumb = str(row.get("thumbnail", "")).strip()
        if thumb in ("nan", "None", ""):
            thumb = ""

        results.append({
            "title":         str(row.get("title",  "Unknown")),
            "author":        str(row.get("author", "Unknown")),
            "genre":         genre_display,
            "description":   str(row.get("description", ""))[:500],
            "rating":        _safe_float(row.get("rating", 0)),
            "ratings_count": _safe_int(row.get("ratings_count", 0)),
            "thumbnail":     thumb,
            "score":         _safe_float(scores[i]),
            "mood":          mood,
        })

    return results
