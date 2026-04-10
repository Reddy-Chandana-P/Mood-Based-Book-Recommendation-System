# Book Recommender

NLP + GenAI powered book recommendation system.

## Setup

### 1. Get a Gemini API Key
Go to https://aistudio.google.com/app/apikey — it's free, no credit card needed.
Add it to `backend/.env`:
```
GEMINI_API_KEY=<your_key>
```

### 2. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
First run will download the dataset and NLP model (~500MB total, cached after that).

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## How it works
- User types how they're feeling (or picks a mood chip)
- NLP (zero-shot classification) detects the mood
- Books are filtered by mood→genre mapping + ranked by TF-IDF similarity + rating
- Gemini explains why each book matches the user's mood
