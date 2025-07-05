import os
import json
import random
import nltk
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://guffgaaf.onrender.com", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load intents globally
try:
    with open("intents.json") as f:
        intents = json.load(f)
except FileNotFoundError:
    intents = {"intents": []}
    print("Error: intents.json not found")

stop_words = set(stopwords.words('english'))

class ChatRequest(BaseModel):
    message: str

def match_intent(user_input):
    if not user_input.strip():
        return "Please enter a valid message!"
    
    tokens = [w.lower() for w in word_tokenize(user_input) if w.lower() not in stop_words and w.isalnum()]
    print(f"Input tokens: {tokens}")  # Debug log
    
    # Check for exact or partial matches
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            pattern_tokens = [w.lower() for w in word_tokenize(pattern) if w.lower() not in stop_words and w.isalnum()]
            if (any(token in pattern_tokens for token in tokens) or 
                user_input.lower() in pattern.lower() or
                any(pattern.lower().startswith(token) for token in tokens)):
                print(f"Matched intent: {intent['tag']}")  # Debug log
                return random.choice(intent["responses"])
    
    print("No intent matched")  # Debug log
    return "Sorry, I didn’t understand. Try asking about workouts or diet!"

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open("static/index.html") as f:
            return f.read()
    except FileNotFoundError:
        return "Error: index.html not found"

@app.head("/")
async def head_root():
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat")
async def chat(request: ChatRequest):
    response = match_intent(request.message)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
