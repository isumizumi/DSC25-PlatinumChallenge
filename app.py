from flask import Flask, request, jsonify
from flasgger import Swagger
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)
swagger = Swagger(app)

# Tokenizer dan model CNN
# with open('tokenizer_cnn.pickle', 'rb') as handle:
#     tokenizer_cnn = pickle.load(handle)
# model_cnn = load_model('model_cnn.h5')

# # Tokenizer dan model RNN
# with open('tokenizer_rnn.pickle', 'rb') as handle:
#     tokenizer_rnn = pickle.load(handle)
# model_rnn = load_model('model_rnn.h5')

# Parameter untuk padding
MAX_SEQUENCE_LENGTH = 36

# Model LSTM dan Tokenizer
model_lstm = load_model('data/model_of_lstm/model-lstm.keras')

tokenizer = Tokenizer()
with open('data/resources_of_lstm/tokenizer-lstm.json', 'r') as f:
    tokenizer_data = f.read()
tokenizer = tokenizer_from_json(tokenizer_data)
print("Tipe Tokenizer:", type(tokenizer))

# Convert text to sequences
def process_text(text, tokenizer, max_sequence_length=36):
    # Mengubah teks menjadi sequences
    sequences = tokenizer.texts_to_sequences([text])
    # Menambahkan padding pada sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    return padded_sequences


label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('data/resources_of_lstm/classes.npy', allow_pickle=True)

# Load slang dictionary
kamus_alay = pd.read_csv('data/new_kamusalay.csv', header=None, index_col=0, encoding='latin-1').squeeze(axis=1).to_dict()

# Load abusive words
abusive_words = set(pd.read_csv('data/abusive.csv', header=None, encoding='latin-1')[0].tolist())

# NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(stopwords.words('indonesian'))

# Fungsi pembersihan teks
def cleansing_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-z\s!?]', '', text)  # Remove special characters except '!?'
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    return text

# Fungsi normalisasi slang
def normalize_slang(text, kamus_alay):
    words = text.split()
    normalized_words = [kamus_alay.get(word, word) for word in words]
    return ' '.join(normalized_words)

# Fungsi untuk menangani abusive words
def handle_abusive_words(text, replacement=''):
    words = text.split()
    handled_words = [replacement if word in abusive_words else word for word in words]
    return ' '.join(handled_words)

# Fungsi untuk menghapus stopwords
def remove_stopwords(text, stop_words):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# @app.route('/predict/cnn', methods=['GET'])
# def predict_cnn():
#     """
#     Prediksi sentimen menggunakan model CNN
#     ---
#     parameters:
#       - name: text
#         in: query
#         type: string
#         required: true
#         description: Teks yang akan diprediksi sentimennya
#     responses:
#       200:
#         description: Hasil prediksi sentimen
#         examples:
#           sentiment: positif
#       400:
#         description: Permintaan tidak valid
#       500:
#         description: Kesalahan server
#     """
#     text = request.args.get('text')
#     if not text:
#         return jsonify({'error': 'Parameter "text" diperlukan.'}), 400

#     try:
#         # Pembersihan teks
#         cleaned_text = cleansing_text(text)
#         # Ekstraksi fitur
#         sequences = tokenizer_cnn.texts_to_sequences([cleaned_text])
#         padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#         # Prediksi
#         prediction = model_cnn.predict(padded_sequences)
#         sentiment = np.argmax(prediction, axis=1)[0]
#         sentiment_label = ['negatif', 'netral', 'positif'][sentiment]
#         return jsonify({'sentiment': sentiment_label}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/predict/rnn', methods=['GET'])
# def predict_rnn():
#     """
#     Prediksi sentimen menggunakan model RNN
#     ---
#     parameters:
#       - name: text
#         in: query
#         type: string
#         required: true
#         description: Teks yang akan diprediksi sentimennya
#     responses:
#       200:
#         description: Hasil prediksi sentimen
#         examples:
#           sentiment: positif
#       400:
#         description: Permintaan tidak valid
#       500:
#         description: Kesalahan server
#     """
#     text = request.args.get('text')
#     if not text:
#         return jsonify({'error': 'Parameter "text" diperlukan.'}), 400

#     try:
#         # Pembersihan teks
#         cleaned_text = cleansing_text(text)
#         # Ekstraksi fitur
#         sequences = tokenizer_rnn.texts_to_sequences([cleaned_text])
#         padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#         # Prediksi
#         prediction = model_rnn.predict(padded_sequences)
#         sentiment = np.argmax(prediction, axis=1)[0]
#         sentiment_label = ['negatif', 'netral', 'positif'][sentiment]
#         return jsonify({'sentiment': sentiment_label}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# Endpoint GET untuk status server
@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'API is running'})

# Endpoint GET untuk friendly message
@app.route('/api/predict', methods=['GET'])
def predict_info():
    return jsonify({
        'message': 'Use POST method to access this endpoint. Send JSON with key "text".',
        'example': {
            'text': 'Your text here'
        }
    })

# Endpoint POST untuk prediksi teks
@app.route('/api/predict/lstm', methods=['POST'])
def predict():
    try:
        # Ambil data dari request
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400

        text = data['text']

        # Proses teks: cleansing, normalisasi slang, dll.
        processed_text = cleansing_text(text)
        processed_text = normalize_slang(processed_text, kamus_alay)
        processed_text = handle_abusive_words(processed_text, abusive_words)
        processed_text = remove_stopwords(processed_text, stop_words)

        # Tokenisasi dan padding
        padded_sequences = process_text(processed_text, tokenizer, max_sequence_length=50)

        # Prediksi
        prediction = model_lstm.predict(padded_sequences)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        confidence = float(np.max(prediction))  # Ambil probabilitas tertinggi

        # Hasil
        result = {
            "text": text,
            "processed_text": processed_text,
            "prediction": predicted_label,
            "confidence": confidence
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
