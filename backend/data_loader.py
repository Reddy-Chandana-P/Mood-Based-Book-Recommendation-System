from __future__ import annotations
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os, urllib.request

CACHE_PATH = "books_cache.pkl"

MOOD_GENRE_MAP = {
    "happy":       ["humor","comedy","romance","children","feel-good","funny","heartwarming",
                    "contemporary","chick-lit","cozy","middle-grade","graphic-novels","light"],
    "sad":         ["self-help","inspirational","memoir","poetry","drama","literary-fiction",
                    "grief","mental-health","personal-development","psychology","healing","emotional"],
    "adventurous": ["adventure","fantasy","science-fiction","thriller","action","epic-fantasy",
                    "space-opera","dystopia","survival","quest","high-fantasy","steampunk","exploration"],
    "romantic":    ["romance","love","chick-lit","contemporary-romance","historical-romance",
                    "paranormal-romance","new-adult","romantic-suspense","love-story"],
    "curious":     ["science","history","biography","non-fiction","philosophy","popular-science",
                    "technology","economics","politics","true-crime","nature","mathematics","psychology"],
    "scared":      ["horror","mystery","thriller","suspense","gothic","dark","psychological-thriller",
                    "crime","supernatural","occult","ghost","serial-killer","noir"],
    "nostalgic":   ["classics","historical-fiction","memoir","literary-fiction","vintage",
                    "19th-century","war","historical","coming-of-age","bildungsroman"],
    "motivated":   ["self-help","business","biography","personal-development","leadership",
                    "entrepreneurship","productivity","success","finance","motivation","sports"],
    "bored":       ["adventure","fantasy","mystery","thriller","action","page-turner",
                    "graphic-novels","short-stories","satire","humor","crime","fast-paced"],
    "relaxed":     ["travel","nature","poetry","short-stories","cozy-mystery","slice-of-life",
                    "food","art","essays","light-fiction","beach-read","gardening"],
    "stressed":    ["self-help","mindfulness","meditation","wellness","cozy","light-fiction",
                    "humor","mental-health","psychology"],
    "inspired":    ["biography","autobiography","inspirational","spirituality","philosophy",
                    "self-help","true-story","leadership","art"],
    "anxious":     ["mindfulness","meditation","self-help","mental-health","psychology",
                    "wellness","anxiety","calm","cozy","humor","light-fiction","poetry"],
    "healing":     ["memoir","self-help","inspirational","poetry","grief","mental-health",
                    "wellness","personal-development","spirituality","healing","recovery"],
}

_REQUIRED = ["title","author","genre","description","rating","ratings_count","thumbnail"]


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns exist with correct types."""
    for col in _REQUIRED:
        if col not in df.columns:
            df[col] = "" if col not in ("rating","ratings_count") else 0
    df["genre"]         = df["genre"].fillna("fiction").astype(str).str.lower().str.strip()
    df["rating"]        = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(0, 5)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
    df["title"]         = df["title"].astype(str).str.strip()
    df["author"]        = df["author"].fillna("Unknown").astype(str).str.strip()
    df["description"]   = df["description"].fillna("").astype(str)
    df["thumbnail"]     = df["thumbnail"].fillna("").astype(str).replace("nan","")
    return df[_REQUIRED].copy()


def _load_goodbooks() -> pd.DataFrame:
    print("  → goodbooks-10k ...")
    books     = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv")
    tags_df   = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv")
    tag_names = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv")

    merged   = tags_df.merge(tag_names, on="tag_id")
    top_tags = (
        merged.sort_values("count", ascending=False)
        .groupby("goodreads_book_id").head(5)
        .groupby("goodreads_book_id")["tag_name"]
        .apply(lambda x: " ".join(x))
        .reset_index().rename(columns={"tag_name": "genre"})
    )
    books = books.merge(top_tags, on="goodreads_book_id", how="left")
    books = books.rename(columns={
        "average_rating": "rating",
        "authors":        "author",
        "image_url":      "thumbnail",
    })
    year = pd.to_numeric(books.get("original_publication_year"), errors="coerce")
    books["description"] = (
        '"' + books["title"].fillna("") + '" by ' + books["author"].fillna("") +
        ". Tags: " + books["genre"].fillna("fiction") +
        year.apply(lambda y: f". Year: {int(y)}" if pd.notna(y) else "")
    )
    print(f"     loaded {len(books)} books")
    return _normalise(books)


def _load_cmu() -> pd.DataFrame:
    """CMU Book Summaries — real plot descriptions."""
    print("  → CMU Book Summaries ...")
    url = "http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz"
    try:
        import tarfile, tempfile, ast
        with urllib.request.urlopen(url, timeout=60) as resp:
            raw = resp.read()
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        rows = []
        with tarfile.open(tmp_path) as tar:
            for member in tar.getmembers():
                if member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if not f:
                        continue
                    for line in f.read().decode("utf-8", errors="ignore").splitlines():
                        parts = line.strip().split("\t")
                        if len(parts) < 6:
                            continue
                        try:
                            genres_raw = ast.literal_eval(parts[5]) if parts[5] else {}
                            genre_str  = " ".join(genres_raw.values()) if isinstance(genres_raw, dict) else ""
                        except Exception:
                            genre_str = ""
                        rows.append({
                            "title":         parts[2],
                            "author":        parts[3],
                            "genre":         genre_str.lower(),
                            "description":   parts[6] if len(parts) > 6 else "",
                            "rating":        0.0,
                            "ratings_count": 0,
                            "thumbnail":     "",
                        })
        os.unlink(tmp_path)
        df = pd.DataFrame(rows)
        df = df[df["description"].astype(str).str.len() > 50].copy()
        print(f"     loaded {len(df)} CMU books")
        return _normalise(df)
    except Exception as e:
        print(f"     CMU load failed ({e}), skipping")
        return pd.DataFrame(columns=_REQUIRED)


def load_books() -> pd.DataFrame:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    print("📥 Building dataset from multiple sources...")
    frames = []

    try:
        gb = _load_goodbooks()
        if not gb.empty:
            frames.append(gb)
    except Exception as e:
        print(f"  goodbooks failed: {e}")

    try:
        cmu = _load_cmu()
        if not cmu.empty:
            frames.append(cmu)
    except Exception as e:
        print(f"  CMU failed: {e}")

    # Google Books — fetch per mood category
    try:
        from google_books import fetch_google_books
        print("  → Google Books API ...")
        gbooks = fetch_google_books()
        if not gbooks.empty:
            frames.append(_normalise(gbooks))
    except Exception as e:
        print(f"  Google Books failed: {e}")

    if not frames:
        raise RuntimeError("All data sources failed. Check your internet connection.")

    df = pd.concat(frames, ignore_index=True)
    df = _normalise(df)

    # Fill short/missing descriptions
    short = df["description"].str.len() < 20
    df.loc[short, "description"] = (
        '"' + df.loc[short, "title"] + '" by ' + df.loc[short, "author"] +
        ". Genre: " + df.loc[short, "genre"]
    )

    # Deduplicate by title + author
    df = df.drop_duplicates(subset=["title", "author"]).reset_index(drop=True)
    print(f"✅ Total unique books: {len(df)}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(df, f)
    return df


def build_tfidf(df: pd.DataFrame):
    content    = (df["description"] + " " + df["genre"].fillna("")).fillna("")
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", ngram_range=(1, 2))
    matrix     = vectorizer.fit_transform(content)
    return vectorizer, matrix


def get_genre_matches(df: pd.DataFrame, mood: str) -> pd.DataFrame:
    genres = MOOD_GENRE_MAP.get(mood.lower(), [])
    if not genres:
        return df
    pattern = "|".join(map(re.escape, genres))
    matched = df[df["genre"].str.contains(pattern, case=False, na=False, regex=True)]
    return matched if len(matched) >= 20 else df
