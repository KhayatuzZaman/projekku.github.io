import io
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, render_template
import re
import json
import random
from urllib import request as urllib_request
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

app = Flask(__name__)

# Load tokenizer and model
class SkinProblemClassifier:
    def __init__(self):
        self.data = self.load_data()
        self.df = self.prepare_data(self.data)

        self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)

        # Mapping categories
        self.category_to_int = {cat: idx for idx, cat in enumerate(self.df['category'].unique())}
        self.int_to_category = {idx: cat for cat, idx in self.category_to_int.items()}

        # Create tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.df['processed_question'])

        X = self.tokenizer.texts_to_sequences(self.df['processed_question'])
        self.max_len = max(len(x) for x in X)

        # Load model
        self.model = tf.keras.models.load_model('models/skin_problem_model.h5')

    def load_data(self):
        url = "https://raw.githubusercontent.com/amalianurislami/Chatbot_Safindra/refs/heads/main/dataset.json"
        response = urllib_request.urlopen(url)
        return json.loads(response.read())

    def preprocess_text(self, text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = ' '.join(text.split())
        return text

    def prepare_data(self, data):
        import pandas as pd
        questions, categories, responses = [], [], []
        for intent in data['masalah_kulit']:
            category = intent['kategori']
            for item in intent['data']:
                questions.append(item['pertanyaan'])
                categories.append(category)
                responses.append(item['jawaban'])

        return pd.DataFrame({
            'question': questions,
            'category': categories,
            'response': responses
        })

    def predict(self, user_input):
        processed = self.preprocess_text(user_input)
        sequence = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(sequence, maxlen=self.max_len)

        prediction = self.model.predict(padded, verbose=0)
        predicted_category_idx = np.argmax(prediction[0])
        predicted_category = self.int_to_category[predicted_category_idx]
        confidence = float(np.max(prediction[0]))

        possible_responses = [item['jawaban'] for intent in self.data['masalah_kulit'] if intent['kategori'] == predicted_category for item in intent['data']]
        response = random.choice(possible_responses) if possible_responses else "Maaf, saya tidak mengerti pertanyaan Anda."
        return {'category': predicted_category, 'confidence': confidence, 'response': response}

    def find_most_similar_question(self, user_input):
        max_similarity = 0
        best_match = None
        best_response = None

        for intent in self.data['masalah_kulit']:
            for item in intent['data']:
                question = item['pertanyaan']
                similarity = SequenceMatcher(None, user_input, question).ratio()
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = question
                    best_response = item['jawaban']

        return best_match, best_response, max_similarity

# Initialize classifier
text_classifier = SkinProblemClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/profil')
def profil():
    return render_template('profil.html')

@app.route('/konsultasi')
def konsultasi():
    return render_template('konsultasi.html')

@app.route('/produk')
def produk():
    return render_template('produk.html')

@app.route('/predict/text', methods=['POST'])
def predict_text():
    user_input = request.json.get('question', '')
    try:
        result = text_classifier.predict(user_input)
        best_match, best_response, similarity = text_classifier.find_most_similar_question(user_input)

        result['best_match'] = best_match
        result['similarity'] = similarity
        if best_response:
            result['response'] = best_response

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
