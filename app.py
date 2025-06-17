
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

# Nonaktifkan GPU dan warning oneDNN
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# Konfigurasi standar NLP
CONFIG = {
    'MIN_WORDS_FOR_SUMMARY': 30,        # Minimal kata untuk summarization
    'MIN_SENTENCES_FOR_SUMMARY': 3,     # Minimal kalimat untuk summarization
    'SUMMARY_SENTENCES': 2,             # Jumlah kalimat dalam ringkasan
    'MAX_SEQUENCE_LENGTH': 100,         # Maksimal panjang sequence
    'NEUTRAL_THRESHOLD': 5.0,           # Batas sentiment netral (+/- %)
    'CONFIDENCE_DECIMALS': 1            # Angka desimal untuk confidence score
}

# Konfigurasi path
MODEL_ID = "1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.h5")
TOKENIZER_PATH = os.path.join("utils", "tokenizer.json")

# Inisialisasi NLTK dengan penanganan error
def init_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError as e:
        print(f"⚠️ Mengunduh data NLTK: {str(e)}")
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        print(f"❌ Error inisialisasi NLTK: {str(e)}")

init_nltk()

# Load tokenizer dengan validasi
def load_tokenizer():
    try:
        with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
            if not isinstance(tokenizer_data, dict):
                raise ValueError("Format tokenizer tidak valid")
            tokenizer = tokenizer_from_json(json.dumps(tokenizer_data))
        print("✅ Tokenizer berhasil dimuat")
        return tokenizer
    except Exception as e:
        print(f"❌ Gagal memuat tokenizer: {str(e)}")
        return None

tokenizer = load_tokenizer()

# Fungsi untuk menentukan perlu summarization atau tidak
def should_summarize(text):
    """Menentukan apakah teks perlu diringkas berdasarkan standar NLP"""
    words = word_tokenize(text)
    sentences = sent_tokenize(text)
    return (len(words) >= CONFIG['MIN_WORDS_FOR_SUMMARY'] and 
            len(sentences) >= CONFIG['MIN_SENTENCES_FOR_SUMMARY'])

# Fungsi summarization dengan standar NLP
def summarize_text(text):
    """Extractive summarization mengikuti best practice NLP"""
    if not should_summarize(text):
        return text
    
    try:
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        # Stopwords untuk bahasa Indonesia yang lebih lengkap
        stop_words = set(stopwords.words('indonesian') + 
                        list(string.punctuation) + 
                        ["yang", "dan", "itu", "dengan", "ini", "dia"])
        
        # Perhitungan frekuensi kata
        word_freq = {}
        for word in words:
            if word not in stop_words and word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Normalisasi frekuensi
        if word_freq:
            max_freq = max(word_freq.values())
            word_freq = {k: v/max_freq for k, v in word_freq.items()}
        
        # Penilaian kalimat
        sentence_scores = {}
        for sent in sentences:
            for word in word_tokenize(sent.lower()):
                if word in word_freq:
                    sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]
        
        # Ambil kalimat terbaik dengan urutan asli
        if sentence_scores:
            ranked_sentences = sorted(sentence_scores.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:CONFIG['SUMMARY_SENTENCES']]
            return ' '.join([s[0] for s in sorted(ranked_sentences, 
                                                key=lambda x: sentences.index(x[0]))])
        return text
    except Exception as e:
        print(f"❌ Error summarization: {str(e)}")
        return text

# Preprocessing teks standar NLP
def preprocess_text(text, tokenizer):
    """Pipeline preprocessing teks standar NLP"""
    try:
        # Pembuatan sequence dengan penanganan OOV
        sequences = tokenizer.texts_to_sequences([text])
        # Padding yang cerdas
        padded = pad_sequences(sequences, 
                             maxlen=CONFIG['MAX_SEQUENCE_LENGTH'],
                             padding='post',
                             truncating='post')
        return padded
    except Exception as e:
        print(f"❌ Error preprocessing: {str(e)}")
        raise

# Download model
def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("🔽 Mengunduh model dari Google Drive...")
        try:
            gdown.download(id=MODEL_ID, output=MODEL_PATH, quiet=False)
            print("✅ Model berhasil diunduh")
            return True
        except Exception as e:
            print(f"❌ Gagal mengunduh model: {str(e)}")
            return False
    return True

# Load model dengan validasi
def load_my_model():
    if not download_model():
        return None
    
    try:
        print("📦 Memuat model dengan validasi...")
        model = keras_load_model(MODEL_PATH)
        # Validasi sederhana
        if not hasattr(model, 'predict'):
            raise ValueError("Format model tidak valid")
        print("✅ Model berhasil dimuat dan divalidasi")
        return model
    except Exception as e:
        print(f"❌ Gagal memuat model: {str(e)}")
        return None

model = load_my_model()

# Endpoint health check
@app.route("/", methods=["GET"])
def health_check():
    status = {
        "status": "ready" if model and tokenizer else "error",
        "model_loaded": bool(model),
        "tokenizer_loaded": bool(tokenizer),
        "summarization": "active"
    }
    return jsonify(status), 200 if model and tokenizer else 503

# Endpoint prediksi utama
@app.route('/predict', methods=['POST'])
def predict():
    if not (model and tokenizer):
        return jsonify({"error": "Layanan belum siap"}), 503
    
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Parameter 'text' diperlukan"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Teks tidak boleh kosong"}), 400
    
    try:
        # Pipeline pemrosesan standar NLP
        summary = summarize_text(text)
        processed = preprocess_text(summary, tokenizer)
        prediction = model.predict(processed)[0]
        
        # Analisis sentimen yang ditingkatkan
        positive = float(prediction[0]) * 100
        negative = 100 - positive
        
        # Deteksi range netral
        if abs(positive - 50) < CONFIG['NEUTRAL_THRESHOLD']:
            sentiment = "Netral"
        elif positive > 50:
            sentiment = "Positif"
        else:
            sentiment = "Negatif"
        
        return jsonify({
            "text": text,
            "summary": summary,
            "sentiment": sentiment,
            "confidence": {
                "positive": round(positive, CONFIG['CONFIDENCE_DECIMALS']),
                "negative": round(negative, CONFIG['CONFIDENCE_DECIMALS'])
            },
            "language": "id"  # Kode bahasa ISO 639-1
        })
    except Exception as e:
        print(f"❌ Error prediksi: {str(e)}")
        return jsonify({
            "error": "Error internal server",
            "details": str(e)[:200]  # Batasi panjang detail error
        }), 500

# Contoh endpoint GET
@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({
        "petunjuk": "Kirim POST request dengan body JSON berisi 'text'",
        "contoh": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "text": "Pelayanan sangat memuaskan!"
            }
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)