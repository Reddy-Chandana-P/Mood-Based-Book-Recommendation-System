import pandas as pd
import numpy as np
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

CACHE_PATH = "books_cache.pkl"

# Public CSV of ~7k books with title, authors, description, categories, rating
DATASET_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"

# Mood to genre mapping
MOOD_GENRE_MAP = {
    "happy": ["humor", "comedy", "romance", "children", "feel-good"],
    "sad": ["self-help", "inspirational", "memoir", "poetry", "drama"],
    "adventurous": ["adventure", "fantasy", "science fiction", "thriller", "action"],
    "romantic": ["romance", "love", "chick lit", "contemporary romance"],
    "curious": ["science", "history", "biography", "non-fiction", "philosophy"],
    "scared": ["horror", "mystery", "thriller", "suspense", "gothic"],
    "nostalgic": ["classics", "historical fiction", "memoir", "literary fiction"],
    "motivated": ["self-help", "business", "biography", "personal development"],
    "bored": ["adventure", "fantasy", "mystery", "thriller", "action"],
    "relaxed": ["travel", "nature", "poetry", "short stories", "cozy mystery"],
}

def load_books() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("Downloading books dataset...")
    # goodbooks-10k: book_id, title, authors, average_rating, ratings_count, image_url
    books_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
    # book tags for genre info
    tags_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv"
    tag_names_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv"

    df = pd.read_csv(books_url)
    tags_df = pd.read_csv(tags_url)
    tag_names_df = pd.read_csv(tag_names_url)

    # Merge tags to get genre-like labels
    tags_merged = tags_df.merge(tag_names_df, on="tag_id")
    # Top tag per book as genre
    top_tags = (
        tags_merged.sort_values("count", ascending=False)
        .groupby("goodreads_book_id")
        .first()
        .reset_index()[["goodreads_book_id", "tag_name"]]
        .rename(columns={"tag_name": "genre"})
    )
    df = df.merge(top_tags, on="goodreads_book_id", how="left")

    # Normalize columns
    df = df.rename(columns={
        "average_rating": "rating",
        "ratings_count": "ratings_count",
        "authors": "author",
        "image_url": "thumbnail",
    })

    df["genre"] = df["genre"].fillna("fiction").str.lower()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)

    # Use title + author as description fallback (no description in this dataset)
    df["description"] = "By " + df["author"].fillna("") + ". Genre: " + df["genre"]

    df = df[["title", "author", "genre", "description", "rating", "ratings_count", "thumbnail"]].dropna(subset=["title"])

    # Sample 6000 for speed
    if len(df) > 6000:
        df = df.sample(6000, random_state=42).reset_index(drop=True)

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(df, f)

    print(f"Loaded {len(df)} books.")
    return df


def build_tfidf(df: pd.DataFrame):
    """Build TF-IDF matrix on description + genre for content-based filtering."""
    df["content"] = df["description"] + " " + df["genre"].fillna("")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    matrix = vectorizer.fit_transform(df["content"])
    return vectorizer, matrix


def get_genre_matches(df: pd.DataFrame, mood: str) -> pd.DataFrame:
    """Filter books by genres mapped to the given mood."""
    genres = MOOD_GENRE_MAP.get(mood.lower(), [])
    if not genres:
        return df
    pattern = "|".join(genres)
    matched = df[df["genre"].str.contains(pattern, case=False, na=False)]
    return matched if len(matched) > 10 else df
