
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
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import os
import gdown
import nltk
import string
import heapq
import json
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Disable GPU and oneDNN warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# Configuration with NLP standards
CONFIG = {
    'MIN_WORDS_FOR_SUMMARY': 30,        # Minimum words to trigger summarization
    'MIN_SENTENCES_FOR_SUMMARY': 3,     # Minimum sentences to trigger summarization
    'SUMMARY_SENTENCES': 2,             # Number of sentences in summary
    'MAX_SEQUENCE_LENGTH': 100,         # For text padding
    'NEUTRAL_THRESHOLD': 5.0,           # Sentiment neutral range (+/- %)
    'CONFIDENCE_DECIMALS': 1            # Decimal places for confidence scores
}

# Path configuration
MODEL_ID = "1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
TOKENIZER_PATH = os.path.join("utils", "tokenizer.json")

# Initialize NLTK with error handling
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError as e:
        print(f"⚠️ Downloading NLTK data: {str(e)}")
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        print(f"❌ NLTK initialization error: {str(e)}")

init_nltk()

# Enhanced tokenizer loading
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            if not isinstance(tokenizer_data, dict):
                raise ValueError("Invalid tokenizer format")
            tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
        print("✅ Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {str(e)}")
        return None

tokenizer = load_tokenizer()

# NLP-Standard Text Summarization
def should_summarize(text):
    """Determine if text needs summarization based on NLP standards"""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return (len(words) >= CONFIG['MIN_WORDS_FOR_SUMMARY'] and 
            len(sentences) >= CONFIG['MIN_SENTENCES_FOR_SUMMARY'])

def summarize_text(text):
    """Extractive summarization following NLP best practices"""
    if not should_summarize(text):
        return text
    
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Enhanced stopwords for Indonesian
        stop_words = set(stopwords.words('indonesian') + 
                    list(string.punctuation) + 
                    ["yang", "dan", "itu", "dengan"]
        
        # Improved frequency calculation
        word_freq = {}
        for word in words:
            if word not in stop_words and word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalize frequencies
        if word_freq:
            max_freq = max(word_freq.values())
            word_freq = {k: v/max_freq for k, v in word_freq.items()}
        
        # Score sentences
        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_freq:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]
        
        # Get top sentences while preserving order
        if sentence_scores:
            ranked_sentences = sorted(sentence_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:CONFIG['SUMMARY_SENTENCES']]
            # Return in original order
            return ' '.join([s[0] for s in sorted(ranked_sentences, 
                                                key=lambda x: sentences.index(x[0]))])
        return text
    except Exception as e:
        print(f"❌ Summarization error: {str(e)}")
        return text

# Enhanced Text Preprocessing
def preprocess_text(text, tokenizer):
    """NLP-standard text preprocessing pipeline"""
    try:
        # Sequence creation with OOV handling
        sequences = tokenizer.texts_to_sequences([text])
        # Smart padding
        padded = pad_sequences(sequences, 
                             maxlen=CONFIG['MAX_SEQUENCE_LENGTH'],
                             padding='post',
                             truncating='post')
        return padded
    except Exception as e:
        print(f"❌ Preprocessing error: {str(e)}")
        raise

# Model loading with validation
def load_my_model():
    if not download_model():
        return None
    
    try:
        print("📦 Loading model with validation...")
        model = keras_load_model(MODEL_PATH)
        # Simple validation
        if not hasattr(model, 'predict'):
            raise ValueError("Invalid model format")
        print("✅ Model loaded and validated")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None

# Enhanced Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if not (model and tokenizer):
        return jsonify({"error": "Service not ready"}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Text parameter required"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400
    
    try:
        # NLP-standard processing pipeline
        summary = summarize_text(text)
        processed = preprocess_text(summary, tokenizer)
        prediction = model.predict(processed)[0]
        
        # Enhanced sentiment analysis
        positive = float(prediction[0]) * 100
        negative = 100 - positive
        
        # Neutral range detection
        if abs(positive - 50) < CONFIG['NEUTRAL_THRESHOLD']:
            sentiment = "Neutral"
        else:
            sentiment = "Positive" if positive >= 50 else "Negative"
        
        return jsonify({
            "text": text,
            "summary": summary,
            "sentiment": sentiment,
            "confidence": {
                "positive": round(positive, CONFIG['CONFIDENCE_DECIMALS']),
                "negative": round(negative, CONFIG['CONFIDENCE_DECIMALS'])
            },
            "language": "id"  # ISO 639-1 language code
        })
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)[:200]  # Limit error details length
        }), 500

        

# Rest of the code remains the same...