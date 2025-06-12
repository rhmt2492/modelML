
# utils/preprocess.py
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Load tokenizer dari file JSON
with open("utils/tokenizer.json", encoding="utf-8") as f:
    tokenizer_json = f.read()  # ✅ baca sebagai string
tokenizer = tokenizer_from_json(tokenizer_json)  # ✅ benar


def preprocess_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)  # Sesuaikan dengan maxlen saat training
    return padded
