import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import random
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from gensim.models import Word2Vec

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('Gym_chatbot_lstm.h5')
word2vec_model = Word2Vec.load('word2vec.model')

conversation_history = []

def get_sentence_embedding(sentence):
    words_in_sentence = [
        lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sentence)
        if w not in ['?', '!', '.', ',', "'s", "'m", "'re", "'ll", "'ve", "'d", "'t", "n't", "``", "''"]
    ]
    if not words_in_sentence:
        return np.zeros(300)
    embeddings = [word2vec_model.wv[w] for w in words_in_sentence if w in word2vec_model.wv]
    if not embeddings:
        return np.zeros(300)
    return np.mean(embeddings, axis=0)

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

def get_response(ints, intents_json):
    if not ints:
        return "Sorry, I didn't understand that. Can you rephrase?"
    
    tag = ints[0]['intent']
    conversation_history.append(tag)

    default_exercises = ['protein shake', 'chicken breast', 'brown rice']
    exercise_intents = ['diet_muscle_gain', 'diet_fat_loss', 'rest_days', 'diet_protein', 'workout_schedule']

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

            # Handle dynamic logic if {exercise} is in response
            if '{exercise}' in response:
                # Determine muscle group
                muscle = tag.split('_')[1] if tag.startswith('workout_') else 'general'
                exercise_list = exercises_map.get(muscle.lower(), default_exercises)
                exercise = random.choice(exercise_list)
                try:
                    return response.format(exercise=exercise)
                except KeyError:
                    return response.replace("{exercise}", exercise)

            # Context-aware reply
            if 'workout_chest' in conversation_history and tag == 'full_body_workout':
                return f"{response} Since you mentioned chest workouts, include pull-ups to balance with back training."

            return response

    return "I’m not sure how to help with that. Try asking about workouts or nutrition!"

# --- CLI Chat Interface ---
print("Chatbot is ready to talk! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print("Chatbot: Goodbye! Stay fit!")
        break
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(f"Chatbot: {res}")
