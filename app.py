from flask import Flask, request, jsonify
from flasgger import Swagger
import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
swagger = Swagger(app)

# Fungsi pembersihan teks
def cleansing_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Tokenizer dan model CNN
# with open('tokenizer_cnn.pickle', 'rb') as handle:
#     tokenizer_cnn = pickle.load(handle)
# model_cnn = load_model('model_cnn.h5')

# # Tokenizer dan model RNN
# with open('tokenizer_rnn.pickle', 'rb') as handle:
#     tokenizer_rnn = pickle.load(handle)
# model_rnn = load_model('model_rnn.h5')

# Tokenizer dan model LSTM
with open('tokenizer_lstm.pickle', 'rb') as handle:
    tokenizer_lstm = pickle.load(handle)
model_lstm = load_model('model_of_lstm/model_lstm.keras')

# Parameter untuk padding
MAX_SEQUENCE_LENGTH = 100

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

@app.route('/predict/lstm', methods=['GET'])
def predict_lstm():
    """
    Prediksi sentimen menggunakan model LSTM
    ---
    parameters:
      - name: text
        in: query
        type: string
        required: true
        description: Teks yang akan diprediksi sentimennya
    responses:
      200:
        description: Hasil prediksi sentimen
        examples:
          sentiment: positif
      400:
        description: Permintaan tidak valid
      500:
        description: Kesalahan server
    """
    text = request.args.get('text')
    if not text:
        return jsonify({'error': 'Parameter "text" diperlukan.'}), 400

    try:
        # Pembersihan teks
        cleaned_text = cleansing_text(text)
        # Ekstraksi fitur
        sequences = tokenizer_lstm.texts_to_sequences([cleaned_text])
        padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        # Prediksi
        prediction = model_lstm.predict(padded_sequences)
        sentiment = np.argmax(prediction, axis=1)[0]
        sentiment_label = ['negatif', 'netral', 'positif'][sentiment]
        return jsonify({'sentiment': sentiment_label}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
