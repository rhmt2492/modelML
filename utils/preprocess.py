
# # utils/preprocess.py
# import json
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# import os

# # Load tokenizer dari file JSON
# with open("utils/tokenizer.json", encoding="utf-8") as f:
#     tokenizer_json = f.read()  # ✅ baca sebagai string
# tokenizer = tokenizer_from_json(tokenizer_json)  # ✅ benar


# def preprocess_text(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=100)  # Sesuaikan dengan maxlen saat training
#     return padded





# import json
# import os
# import gdown
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# # Path tokenizer
# tokenizer_path = os.path.join("utils", "tokenizer.json")

# # Unduh tokenizer dari Google Drive jika belum ada
# if not os.path.exists(tokenizer_path):
#     os.makedirs("utils", exist_ok=True)
#     print("🔽 Downloading tokenizer from Google Drive...")
#     gdown.download(id="1SIAsy1-qAjmN2TRedi6tVtS1TQezc-2K", output=tokenizer_path, quiet=False)

# # Load tokenizer dari file JSON
# with open(tokenizer_path, encoding="utf-8") as f:
#     tokenizer_json = f.read()
# tokenizer = tokenizer_from_json(tokenizer_json)

# # Fungsi preprocessing
# def preprocess_text(text):
#     seq = tokenizer.texts_to_sequences([text])
#     padded = pad_sequences(seq, maxlen=100)  # Sesuaikan dengan saat training
#     return padded



# utils/preprocess.py
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Load tokenizer dari file JSON
tokenizer_path = "utils/tokenizer.json"
try:
    print("📥 Loading tokenizer...")
    with open(tokenizer_path, encoding="utf-8") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    print("✅ Tokenizer loaded.")
except Exception as e:
    print("❌ Failed to load tokenizer:", e)
    tokenizer = None

def preprocess_text(text):
    if tokenizer is None:
        raise ValueError("Tokenizer not available")
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    return padded
