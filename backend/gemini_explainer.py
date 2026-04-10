import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-1.5-flash")


def explain_recommendation(book: dict, user_text: str, mood: str) -> str:
    prompt = f"""You are a warm, knowledgeable book curator helping someone find their next great read.

User's message: "{user_text}"
Detected mood: {mood}

Book details:
- Title: {book['title']}
- Author: {book['author']}
- Genre/Tags: {book['genre']}
- Rating: {book['rating']}/5 ({book.get('ratings_count', 0):,} ratings)

Write a 3-4 sentence personalized recommendation explaining:
1. Why this book perfectly matches their current mood
2. What makes this book special or unique
3. What emotional experience or takeaway they can expect

Be conversational, enthusiastic but not over the top. Avoid generic phrases like "this book is perfect for you". Make it feel like advice from a friend who has read it.
"""
    try:
        response = _model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return (
            f"A compelling {mood} read — \"{book['title']}\" by {book['author']} "
            f"has captivated {book.get('ratings_count', 0):,} readers with a {book['rating']}/5 rating. "
            f"Genre: {book['genre']}."
        )
