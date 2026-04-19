import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-1.5-flash")


def explain_recommendation(book: dict, user_text: str, mood: str) -> str:
    title   = book.get("title", "")
    author  = book.get("author", "")
    genre   = book.get("genre", "")
    desc    = book.get("description", "")[:300]
    rating  = book.get("rating", 0)
    count   = book.get("ratings_count", 0)

    prompt = f"""You are a well-read friend giving a genuine, specific book recommendation.

The person said: "{user_text}"
Their mood: {mood}

Book: "{title}" by {author}
Genre: {genre}
Description snippet: {desc}
Rating: {rating}/5 from {count:,} readers

Write 2-3 sentences that feel personal and specific to THIS book — not generic praise.
Focus on one concrete thing about this specific book (a character, a theme, a feeling it creates, what makes it stand out from others in its genre) that directly connects to what the person said.
Do NOT use phrases like "perfect for your mood", "great pick", "this book is for you", or "I highly recommend".
Sound like a friend who actually read it, not a review bot.
Be direct, warm, and specific."""

    try:
        response = _model.generate_content(
            prompt,
            generation_config={"temperature": 0.9, "max_output_tokens": 150}
        )
        return response.text.strip()
    except Exception:
        # Fallback uses the description snippet so it's at least book-specific
        return f'"{title}" has drawn {count:,} readers with a {rating}/5 rating — {desc[:120]}...' if desc else f'A {genre} read by {author}, rated {rating}/5.'
