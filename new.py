import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import numpy as np
import nltk
import json
import random
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# Prepare data with augmentation
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',', "'s", "'m", "'re", "'ll", "'ve", "'d", "'t", "n't", "``", "''"]

# Augment patterns with synonyms
augmented_patterns = {
    'diet_muscle_gain': ['how to build muscle', 'build muscle', 'gain muscle mass', 'muscle growth', 'how to get bigger muscles'],
    'workout_abs': ['abs exercises', 'ab workout', 'core exercises', 'how to train abs', 'abs building'],
    'diet_protein': ['protein intake', 'how much protein', 'protein per day', 'protein needs', 'how much protein should I take'],
    'workout_schedule': ['workout schedule', 'training plan', 'weekly workout plan', 'exercise routine', 'gym schedule']
}

for intent in intents['intents']:
    patterns = intent['patterns']
    tag = intent['tag']
    # Add augmented patterns if available
    if tag in augmented_patterns:
        patterns.extend(augmented_patterns[tag])
    for pattern in patterns:
        word_list = word_tokenize(pattern.lower())
        words.extend(word_list)
        documents.append((" ".join(word_list), tag))
        if tag not in classes:
            classes.append(tag)

words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"Classes: {len(classes)} intents")
print(f"Words: {len(words)} unique words")

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Train Word2Vec model with larger vector size
sentences = [word_tokenize(doc[0].lower()) for doc in documents]
word2vec_model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=1, workers=4)
word2vec_model.save('word2vec.model')

def get_sentence_embedding(sentence):
    words = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(sentence) if w not in ignore_words]
    if not words:
        return np.zeros(300)
    embeddings = [word2vec_model.wv[w] for w in words if w in word2vec_model.wv]
    if not embeddings:
        return np.zeros(300)
    return np.mean(embeddings, axis=0)

# Prepare training data
training = []
for doc in documents:
    embedding = get_sentence_embedding(doc[0])
    output_row = [0] * len(classes)
    output_row[classes.index(doc[1])] = 1
    training.append([embedding, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Enhanced LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(300,)),
    tf.keras.layers.Reshape((1, 300)),
    tf.keras.layers.LSTM(256, return_sequences=True),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=8, verbose=1,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
model.save('Gym_chatbot_lstm.h5')
print("LSTM model trained and saved as 'Gym_chatbot_lstm.h5'")
