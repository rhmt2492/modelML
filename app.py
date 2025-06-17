
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

# 🔽 Tambahan untuk ringkasan teks
import nltk
import string
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download resource NLTK
nltk.download('punkt')
nltk.download('stopwords')

# ==== Fungsi ringkasan ====
def summarize_text(text, num_sentences=2):
    sentences = sent_tokenize(text, language='english')
    words = word_tokenize(text.lower(), language='english')
    stop_words = set(stopwords.words('indonesian'))

    word_frequencies = {}
    for word in words:
        if word not in stop_words and word not in string.punctuation:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower(), language='english'):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

# ==== Flask App ====
app = Flask(__name__)
CORS(app)

# ==== Download model dari Google Drive jika belum ada ====
model_path = os.path.join("model", "sentiment_model.h5")
if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    print("🔽 Downloading model from Google Drive...")
    try:
        gdown.download(id="1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1", output=model_path, quiet=False)
    except Exception as e:
        print("❌ Failed to download model:", e)

# ==== Load model ====
try:
    print("📦 Loading model...")
    model = load_model(model_path)
    print("✅ Model loaded.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# ==== Health check endpoint ====
@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# ==== Predict endpoint (POST) ====
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    text = data['text']
    try:
        # Tambahkan ringkasan
        summary = summarize_text(text)

        # Gunakan ringkasan untuk prediksi
        processed = preprocess_text(summary)
        prediction = model.predict(processed)[0]
        percent_positif = float(prediction[0]) * 100
        percent_negatif = 100 - percent_positif
        label = "Positif" if percent_positif >= 50 else "Negatif"

        return jsonify({
            'text': text,
            'summary': summary,
            'sentiment': label,
            'score': {
                'positif': round(percent_positif, 2),
                'negatif': round(percent_negatif, 2)
            }
        })
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({'error': 'Internal Server Error'}), 500

# ==== Predict endpoint (GET info) ====
@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({
        "message": "Gunakan metode POST dengan JSON body berisi field 'text' untuk melakukan prediksi.",
        "example": {
            "method": "POST",
            "url": "/predict",
            "body": {
                "text": "saya sangat senang dengan pelayanan ini"
            }
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

