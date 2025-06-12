
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# import json
# import os

# # ====== 1. DATA LATIH BARU (POS + NEG) ======
# texts = [
#     # Positif
#     "cerita ini sangat bagus",
#     "saya sangat menyukai pengalaman ini",
#     "kegiatan ini menyenangkan dan mendidik",
#     "aku senang sekali ikut acara ini",
#     "menarik dan penuh wawasan",
#     "alur cerita sangat menyentuh dan menginspirasi",
#     "karakter dalam cerita berkembang dengan sangat baik",
#     "bahasanya indah dan mudah dipahami",
#     "ceritanya membawa pesan moral yang dalam",
#     "ending-nya sangat memuaskan dan tidak terduga",
#     "saya menikmati setiap bagian dari cerita ini",
#     "tokohnya terasa sangat nyata",
#     "cerita ini membuat saya berpikir tentang hidup",
#     "dialog antartokohnya terasa natural dan hidup",
#     "gaya penceritaannya sangat mengalir",
#     "saya merasa terhibur dan terinspirasi oleh cerita ini",
#     "alur ceritanya sangat menarik dan menginspirasi",
#     "saya sangat suka dengan kegiatan ini dan akan terus ikut",
#     "modul ini sangat membantu dalam mengembangkan kemampuan saya",
#     "saya sangat puas dengan hasil latihan ini",

#     # Negatif
#     "cerita ini membosankan dan jelek",
#     "pengalaman ini buruk sekali",
#     "sangat mengecewakan dan tidak direkomendasikan",
#     "aku sangat kecewa dengan kegiatan ini",
#     "tidak menarik dan membosankan",
#     "alur ceritanya sangat membosankan dan monoton",
#     "tokohnya tidak memiliki karakter yang kuat",
#     "ending ceritanya sangat mengecewakan",
#     "bahasanya terlalu bertele-tele dan membingungkan",
#     "saya merasa buang-buang waktu membaca ini",
#     "tidak ada hal menarik di sepanjang cerita",
#     "narasi terlalu datar dan tidak membangkitkan rasa ingin tahu",
#     "saya tidak merasakan keterlibatan emosi sama sekali",
#     "ceritanya terasa sangat dibuat-buat",
#     "saya tidak merekomendasikan cerita ini kepada siapa pun"
#     "alur membosankan, bahasa kaku, tidak ada nilai, hanya buang-buang waktu",
#     "alur terlalu lambat dan tidak jelas, membuat saya kehilangan minat",
#     "cerita ini benar-benar mengecewakan dari awal hingga akhir",
#     "bahasa yang digunakan tidak mengalir dan membuat bosan",

# ]
# labels = [
#     1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1,
#     1, 1, 1, 1, 1,
#     0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0,
#     0, 0, 0, 0, 0
# ]  # 1 = Positif, 0 = Negatif

# # ====== 2. TOKENIZER ======
# tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# tokenizer.fit_on_texts(texts)

# # Simpan tokenizer ke file JSON
# os.makedirs("utils", exist_ok=True)
# tokenizer_json = tokenizer.to_json()
# with open("utils/tokenizer.json", "w", encoding="utf-8") as f:
#     f.write(tokenizer_json)

# # ====== 3. PERSIAPAN DATA ======
# sequences = tokenizer.texts_to_sequences(texts)
# padded_sequences = pad_sequences(sequences, padding='post', maxlen=20)
# labels = np.array(labels)

# # ====== 4. MODEL ======
# model = Sequential([
#     Embedding(input_dim=10000, output_dim=16, input_length=20),
#     LSTM(32),
#     Dense(1, activation='sigmoid')
# ])

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()

# # ====== 5. LATIH MODEL ======
# model.fit(padded_sequences, labels, epochs=10, verbose=1)

# # ====== 6. SIMPAN MODEL ======
# os.makedirs("model", exist_ok=True)
# model.save("model/sentiment_model.h5")
# print("✅ Model dan tokenizer berhasil disimpan!")


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import json
import os

# ====== 1. DATA LATIH BARU (POS + NEG) ======
texts = [
    # Positif
    "cerita ini sangat bagus",
    "saya sangat menyukai pengalaman ini",
    "kegiatan ini menyenangkan dan mendidik",
    "aku senang sekali ikut acara ini",
    "menarik dan penuh wawasan",
    "alur cerita sangat menyentuh dan menginspirasi",
    "karakter dalam cerita berkembang dengan sangat baik",
    "bahasanya indah dan mudah dipahami",
    "ceritanya membawa pesan moral yang dalam",
    "ending-nya sangat memuaskan dan tidak terduga",
    "saya menikmati setiap bagian dari cerita ini",
    "tokohnya terasa sangat nyata",
    "cerita ini membuat saya berpikir tentang hidup",
    "dialog antartokohnya terasa natural dan hidup",
    "gaya penceritaannya sangat mengalir",
    "saya merasa terhibur dan terinspirasi oleh cerita ini",
    "alur ceritanya sangat menarik dan menginspirasi",
    "saya sangat suka dengan kegiatan ini dan akan terus ikut",
    "modul ini sangat membantu dalam mengembangkan kemampuan saya",
    "saya sangat puas dengan hasil latihan ini",

    # Negatif
    "cerita ini membosankan dan jelek",
    "pengalaman ini buruk sekali",
    "sangat mengecewakan dan tidak direkomendasikan",
    "aku sangat kecewa dengan kegiatan ini",
    "tidak menarik dan membosankan",
    "alur ceritanya sangat membosankan dan monoton",
    "tokohnya tidak memiliki karakter yang kuat",
    "ending ceritanya sangat mengecewakan",
    "bahasanya terlalu bertele-tele dan membingungkan",
    "saya merasa buang-buang waktu membaca ini",
    "tidak ada hal menarik di sepanjang cerita",
    "narasi terlalu datar dan tidak membangkitkan rasa ingin tahu",
    "saya tidak merasakan keterlibatan emosi sama sekali",
    "ceritanya terasa sangat dibuat-buat",
    "saya tidak merekomendasikan cerita ini kepada siapa pun",
    "alur membosankan, bahasa kaku, tidak ada nilai, hanya buang-buang waktu",
    "alur terlalu lambat dan tidak jelas, membuat saya kehilangan minat",
    "cerita ini benar-benar mengecewakan dari awal hingga akhir",
    "bahasa yang digunakan tidak mengalir dan membuat bosan",
]

labels = [1] * 20 + [0] * 18  # 1 = Positif, 0 = Negatif

# ====== 2. TOKENIZER ======
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Simpan tokenizer ke file JSON di folder model/
os.makedirs("utils", exist_ok=True)
tokenizer_json = tokenizer.to_json()
with open("utils/tokenizer.json", "w", encoding="utf-8") as f:
    f.write(tokenizer_json)

# ====== 3. PERSIAPAN DATA ======
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post', maxlen=20)
labels = np.array(labels)

# ====== 4. MODEL ======
model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=20),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# ====== 5. LATIH MODEL ======
model.fit(padded_sequences, labels, epochs=10, verbose=1)

# ====== 6. SIMPAN MODEL ======
model.save("model/sentiment_model.h5")
print("✅ Model dan tokenizer berhasil disimpan di folder 'model/'")
