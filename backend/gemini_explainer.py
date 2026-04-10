import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
_model = genai.GenerativeModel("gemini-1.5-flash")

def explain_recommendation(book: dict, user_text: str, mood: str) -> str:
    """Ask Gemini to explain why this book suits the user's mood/input."""
    prompt = f"""
A user said: "{user_text}"
Their detected mood is: {mood}

Recommend the book "{book['title']}" by {book['author']} (Genre: {book['genre']}).
In 2-3 sentences, explain in a warm and engaging tone why this book is a great match for their current mood and what they can expect from it.
Keep it concise and conversational.
"""
    try:
        response = _model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"A great pick for your {mood} mood — {book['description'][:200]}..."
