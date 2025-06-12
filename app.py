
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from utils.preprocess import preprocess_text
# import os

# app = Flask(__name__)

# # Load model
# model_path = os.path.join("model", "sentiment_model.h5")
# model = load_model(model_path)


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'Text is required'}), 400

#     text = data['text']
#     processed = preprocess_text(text)
#     prediction = model.predict(processed)[0]  # Output: satu angka, misal [0.74]

#     percent_positif = float(prediction[0]) * 100
#     percent_negatif = 100 - percent_positif
#     label = "Positif" if percent_positif >= 50 else "Negatif"

#     return jsonify({
#         'text': text,
#         'sentiment': label,
#         'score': {
#             'positif': percent_positif,
#             'negatif': percent_negatif
#         }
#     })

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# from utils.preprocess import preprocess_text
# import os
# import gdown  # pastikan gdown ada di requirements.txt

# app = Flask(__name__)

# # ==== Download model dari Google Drive jika belum ada ====
# model_path = os.path.join("model", "sentiment_model.h5")
# if not os.path.exists(model_path):
#     os.makedirs("model", exist_ok=True)
#     print("🔽 Downloading model from Google Drive...")
#     # Gunakan ID file Google Drive
#     gdown.download(id="1aUMAH8vYY8Qx_efOtKIiUBfU6i6Oa1P1", output=model_path, quiet=False)

# # ==== Load model ====
# model = load_model(model_path)

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'Text is required'}), 400

#     text = data['text']
#     processed = preprocess_text(text)
#     prediction = model.predict(processed)[0]  # Output: satu angka, misal [0.74]

#     percent_positif = float(prediction[0]) * 100
#     percent_negatif = 100 - percent_positif
#     label = "Positif" if percent_positif >= 50 else "Negatif"

#     return jsonify({
#         'text': text,
#         'sentiment': label,
#         'score': {
#             'positif': percent_positif,
#             'negatif': percent_negatif
#         }
#     })

# # if __name__ == '__main__':
# #     app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)



from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from utils.preprocess import preprocess_text
import os
import gdown

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 503

    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Text is required'}), 400

    text = data['text']
    try:
        processed = preprocess_text(text)
        prediction = model.predict(processed)[0]  # Output: satu angka, misal [0.74]
        percent_positif = float(prediction[0]) * 100
        percent_negatif = 100 - percent_positif
        label = "Positif" if percent_positif >= 50 else "Negatif"

        return jsonify({
            'text': text,
            'sentiment': label,
            'score': {
                'positif': percent_positif,
                'negatif': percent_negatif
            }
        })
    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
