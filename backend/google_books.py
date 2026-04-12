from __future__ import annotations
import requests
import time
import pandas as pd

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

# One search query per mood — tuned for best results
MOOD_QUERIES = {
    "happy":       ["funny fiction", "feel good novel"],
    "sad":         ["emotional memoir", "inspirational self help"],
    "adventurous": ["epic fantasy adventure", "science fiction space"],
    "romantic":    ["romance novel", "historical romance"],
    "curious":     ["popular science", "history nonfiction"],
    "scared":      ["horror fiction", "psychological thriller"],
    "nostalgic":   ["classic literature", "historical fiction"],
    "motivated":   ["self help success", "biography entrepreneur"],
    "bored":       ["page turner thriller", "action adventure fiction"],
    "relaxed":     ["cozy mystery", "travel memoir"],
    "stressed":    ["mindfulness meditation", "self help anxiety"],
    "inspired":    ["inspirational biography", "spiritual philosophy"],
    "anxious":     ["anxiety self help", "mindfulness calm"],
    "healing":     ["grief healing memoir", "recovery self help"],
}

MAX_PER_QUERY = 40   # Google Books returns max 40 per request


def _fetch_query(query: str, max_results: int = MAX_PER_QUERY) -> list[dict]:
    """Fetch books from Google Books API for a single query."""
    rows = []
    start = 0
    while start < max_results:
        batch = min(40, max_results - start)
        try:
            resp = requests.get(GOOGLE_BOOKS_URL, params={
                "q":          query,
                "maxResults": batch,
                "startIndex": start,
                "printType":  "books",
                "langRestrict": "en",
            }, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                info = item.get("volumeInfo", {})
                title  = info.get("title", "").strip()
                if not title:
                    continue
                authors     = info.get("authors", ["Unknown"])
                description = info.get("description", "")
                categories  = info.get("categories", [])
                rating      = info.get("averageRating", 0)
                count       = info.get("ratingsCount", 0)
                thumb       = info.get("imageLinks", {}).get("thumbnail", "")
                thumb       = thumb.replace("http://", "https://")  # force HTTPS
                genre       = " ".join(c.lower() for c in categories)

                rows.append({
                    "title":         title,
                    "author":        ", ".join(authors),
                    "genre":         genre or query.lower(),
                    "description":   description[:600] if description else f'"{title}" by {", ".join(authors)}. Genre: {genre or query}',
                    "rating":        float(rating) if rating else 0.0,
                    "ratings_count": int(count) if count else 0,
                    "thumbnail":     thumb,
                })
            start += len(items)
            if len(items) < batch:
                break
            time.sleep(0.1)  # be polite to the API
        except Exception:
            break
    return rows


def fetch_google_books(moods: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch books from Google Books API for all moods (or a subset).
    Returns a DataFrame with the same schema as the main dataset.
    """
    queries = MOOD_QUERIES
    if moods:
        queries = {m: MOOD_QUERIES[m] for m in moods if m in MOOD_QUERIES}

    all_rows = []
    total_queries = sum(len(v) for v in queries.values())
    done = 0

    for mood, query_list in queries.items():
        for query in query_list:
            done += 1
            print(f"  [{done}/{total_queries}] Google Books: '{query}'")
            rows = _fetch_query(query, MAX_PER_QUERY)
            all_rows.extend(rows)
            time.sleep(0.2)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["rating"]        = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(0, 5)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    df = df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)
    print(f"  ✅ Google Books: {len(df)} unique books fetched")
    return df
