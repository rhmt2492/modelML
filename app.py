
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # ⬅️ Tambahkan ini untuk mengaktifkan CORS
# from tensorflow.keras.models import load_model
# from utils.preprocess import preprocess_text
# import os
# import gdown

# app = Flask(__name__)
# CORS(app)  # ⬅️ Aktifkan CORS di seluruh endpoint

# # ==== Download model dari Google Drive jika belum ada ====
# model_path = os.path.join("model", "sentiment_model.h5")
# if not os.path.exists(model_path):
#     os.makedirs("model", exist_ok=True)
#     print("🔽 Downloading model from Google Drive...")
#     try:
#         gdown.download(id="1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1", output=model_path, quiet=False)
#     except Exception as e:
#         print("❌ Failed to download model:", e)

# # ==== Load model ====
# try:
#     print("📦 Loading model...")
#     model = load_model(model_path)
#     print("✅ Model loaded.")
# except Exception as e:
#     print("❌ Error loading model:", e)
#     model = None

# # ==== Health check endpoint ====
# @app.route("/", methods=["GET"])
# def health():
#     return jsonify({"status": "ok"}), 200

# # ==== Predict endpoint (POST) ====
# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({"error": "Model not available"}), 503

#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'Text is required'}), 400

#     text = data['text']
#     try:
#         processed = preprocess_text(text)
#         prediction = model.predict(processed)[0]  # Output: satu angka, misal [0.74]
#         percent_positif = float(prediction[0]) * 100
#         percent_negatif = 100 - percent_positif
#         label = "Positif" if percent_positif >= 50 else "Negatif"

#         return jsonify({
#             'text': text,
#             'sentiment': label,
#             'score': {
#                 'positif': percent_positif,
#                 'negatif': percent_negatif
#             }
#         })
#     except Exception as e:
#         print("❌ Error during prediction:", e)
#         return jsonify({'error': 'Internal Server Error'}), 500

# # ==== Predict endpoint (GET info) ====
# @app.route('/predict', methods=['GET'])
# def predict_get():
#     return jsonify({
#         "message": "Gunakan metode POST dengan JSON body berisi field 'text' untuk melakukan prediksi.",
#         "example": {
#             "method": "POST",
#             "url": "/predict",
#             "body": {
#                 "text": "saya sangat senang dengan pelayanan ini"
#             }
#         }
#     }), 200

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)




from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_text
import os
import gdown
import nltk
import string
import heapq
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ===== Configuration =====
MODEL_ID = "1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
TOKENIZER_PATH = os.path.join("utils", "tokenizer.json")

# ===== Initialize NLTK =====
def initialize_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

initialize_nltk()

# ===== Load Tokenizer =====
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_data = f.read()
            tokenizer = tokenizer_from_json(tokenizer_data)
        print("✅ Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {str(e)}")
        return None

tokenizer = load_tokenizer()

# ===== Text Summarization =====
def summarize_text(text, num_sentences=2):
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('indonesian') + list(string.punctuation))

        word_frequencies = {}
        for word in words:
            if word not in stop_words:
                word_frequencies[word] = word_frequencies.get(word, 0) + 1

        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

        summary_sentences = heapq.nlargest(
            num_sentences, 
            sentence_scores, 
            key=sentence_scores.get
        )
        return ' '.join(summary_sentences)
    except Exception as e:
        print(f"❌ Error in summarization: {str(e)}")
        return text  # Return original text if summarization fails

# ===== Download Model =====
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("🔽 Downloading model from Google Drive...")
        try:
            gdown.download(
                id=MODEL_ID,
                output=MODEL_PATH,
                quiet=False
            )
            print("✅ Model downloaded successfully.")
        except Exception as e:
            print(f"❌ Failed to download model: {str(e)}")
            return False
    return True

# ===== Load Model =====
def load_ml_model():
    if not download_model():
        return None
    
    try:
        print("📦 Loading model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None

model = load_ml_model()

# ===== Helper Functions =====
def is_service_ready():
    return model is not None and tokenizer is not None

# ===== API Endpoints =====
@app.route("/", methods=["GET"])
def health_check():
    status = {
        "status": "ready" if is_service_ready() else "initializing",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }
    status_code = 200 if is_service_ready() else 503
    return jsonify(status), status_code

@app.route('/predict', methods=['POST'])
def predict():
    if not is_service_ready():
        return jsonify({"error": "Service is not ready"}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Text is required in JSON body'}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Text cannot be empty'}), 400

    try:
        # Process the text
        summary = summarize_text(text)
        processed = preprocess_text(summary, tokenizer)
        
        # Make prediction
        prediction = model.predict(processed)[0]
        percent_positive = float(prediction[0]) * 100
        percent_negative = 100 - percent_positive
        label = "Positive" if percent_positive >= 50 else "Negative"

        return jsonify({
            'text': text,
            'summary': summary,
            'sentiment': label,
            'confidence': {
                'positive': round(percent_positive, 2),
                'negative': round(percent_negative, 2)
            }
        })
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({
        "message": "Please use POST method with JSON body containing 'text' field",
        "example": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "text": "Saya sangat senang dengan pelayanan ini"
            }
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)