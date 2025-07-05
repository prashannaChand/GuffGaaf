import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
import pickle
import random
import json
from gensim.models import Word2Vec

# ----- FastAPI Setup -----
app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace * with actual domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Request Schema -----
class Message(BaseModel):
    message: str

# ----- Load Resources -----
lemmatizer = WordNetLemmatizer()

try:
    intents = json.load(open("intents.json"))
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = tf.keras.models.load_model("Gym_chatbot_lstm.h5")
    word2vec_model = Word2Vec.load("word2vec.model")
except Exception as e:
    print(f"[ERROR LOADING MODELS/FILES] {e}")
    raise

conversation_history = []

# ----- Helper Functions -----
def get_sentence_embedding(sentence):
    try:
        tokens = word_tokenize(sentence)
        words_in_sentence = [
            lemmatizer.lemmatize(w.lower()) for w in tokens if w.isalnum()
        ]

        if not words_in_sentence:
            return np.zeros(300)

        embeddings = [word2vec_model.wv[w] for w in words_in_sentence if w in word2vec_model.wv]

        if not embeddings:
            return np.zeros(300)

        return np.mean(embeddings, axis=0)

    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.zeros(300)

def predict_class(sentence):
    embedding = get_sentence_embedding(sentence)
    res = model.predict(np.array([embedding]))[0]
    ERROR_THRESHOLD = 0.05
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    if not results:
        max_index = np.argmax(res)
        results = [[max_index, res[max_index]]]

    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[i[0]], "probability": str(i[1])} for i in results]

def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"

    tag = intents_list[0]['intent']
    conversation_history.append(tag)

    default_exercises = ['protein shake', 'chicken breast', 'brown rice']
    exercises_map = {
        'chest': ['bench press', 'push-ups', 'cable crossovers', 'pec-dec fly', 'chest dips', 'incline press'],
        'arms': ['bicep curls', 'tricep dips', 'hammer curls', 'concentration curls', 'skull crushers'],
        'back': ['pull-ups', 'deadlifts', 'bent-over rows', 'seated rows', 'lat pulldown'],
        'shoulders': ['overhead press', 'lateral raises', 'Arnold press', 'front raises', 'rear delt fly'],
        'legs': ['squats', 'lunges', 'calf raises', 'leg press', 'step-ups', 'Romanian deadlifts'],
        'abs': ['planks', 'crunches', 'Russian twists', 'mountain climbers', 'leg raises'],
        'general': default_exercises
    }

    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])

            if '{exercise}' in response:
                muscle = tag.split('_')[1] if tag.startswith('workout_') else 'general'
                exercise_list = exercises_map.get(muscle.lower(), default_exercises)
                exercise = random.choice(exercise_list)
                return response.replace("{exercise}", exercise)

            if 'workout_chest' in conversation_history and tag == 'full_body_workout':
                return f"{response} Since you mentioned chest workouts, include pull-ups to balance with back training."

            return response

    return "I’m not sure how to help with that. Try asking about workouts or nutrition!"

# ----- API Endpoint -----
@app.post("/chat")
def chat_endpoint(msg: Message):
    intents_list = predict_class(msg.message)
    response = get_response(intents_list, intents)
    return {"response": response}

# ----- Serve UI -----
@app.get("/")
def serve_index():
    return FileResponse("index.html")

# ----- Run app on Render -----
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000)
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
