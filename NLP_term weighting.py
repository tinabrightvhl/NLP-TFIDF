# -*- coding: utf-8 -*-
"""
Term Weighting & TF-IDF dari Nol
================================
Penulis : Wetina (NIM 230840120)
Tugas   : NLP 4 - Term Weighting dan TF-IDF

Deskripsi
---------
Implementasi manual pembobotan term pada Information Retrieval / NLP,
mulai dari Bag of Words sampai TF-IDF, menggunakan 5 dokumen berbahasa
Indonesia bertema wisata. Hasil perhitungan diverifikasi dengan scikit-learn.

Isi program
-----------
1. Bag of Words : urutan kata diabaikan
2. TF (Term Frequency) : matriks term x dokumen
3. Pembobotan biner : ada (1) / tidak ada (0)
4. Log TF : 1 + log10(tf), meredam lonjakan frekuensi
5. DF dan IDF : idf = log10(N / df), semakin langka semakin besar
6. TF-IDF : (1 + log10(tf)) x log10(N / df)
7. Visualisasi bobot term pada d1 dan verifikasi rumus manual
8. Verifikasi dengan CountVectorizer dan TfidfVectorizer (scikit-learn)

Catatan
-------
- scikit-learn memakai log natural (ln), smoothing IDF, dan normalisasi L2,
  sehingga TF-IDF bawaannya berbeda dari hitungan manual (log10).
- Library: pandas, numpy, matplotlib, scikit-learn
- Awalnya dibuat di Google Colab.
"""

# Jalankan di terminal / Colab bila library belum terpasang:
# pip install pandas numpy matplotlib scikit-learn

import math
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

docs = {
    "d1": "Pantai wisata memiliki pasir putih. Wisata pantai dan laut menarik pengunjung.",
    "d2": "Pendaki menuju gunung melalui jalur pendakian. Jalur gunung cukup menantang.",
    "d3": "Hotel wisata menyediakan kamar nyaman. Kamar hotel dekat pantai.",
    "d4": "Museum sejarah menjual tiket masuk. Tiket museum mengenalkan sejarah dan wisata.",
    "d5": "Hutan gunung memiliki udara segar. Jalur hutan menjaga udara tetap bersih."
}

for d, teks in docs.items():
    print(d, ":", teks)

D1 = "pantai wisata pasir"
D2 = "pasir pantai wisata"

print(D1.split())
print(D2.split())
print("Urutan sama?     ", D1.split() == D2.split())
print("Indeks (BoW) sama?", Counter(D1.split()) == Counter(D2.split()))

token = {
    "d1": ["pantai"] * 3 + ["wisata"] * 2 + ["pasir"] * 2 + ["laut"],
    "d2": ["gunung"] * 2 + ["wisata"] + ["jalur"] * 2 + ["pendaki"],
    "d3": ["hotel"] * 2 + ["wisata"] + ["kamar"] * 3 + ["pantai"],
    "d4": ["tiket"] * 2 + ["museum"] * 3 + ["sejarah"] * 2 + ["wisata"],
    "d5": ["gunung"] + ["hutan"] * 2 + ["jalur"] + ["udara"] * 2
}

for d, ts in token.items():
    print(d, sorted(Counter(ts).items()), "| jumlah token:", len(ts))

urutan = [
    "pantai", "wisata", "pasir", "laut", "gunung", "jalur", "pendaki",
    "hotel", "kamar", "tiket", "museum", "sejarah", "hutan", "udara"
]

tf = pd.DataFrame({d: Counter(ts) for d, ts in token.items()})
tf = tf.reindex(urutan).fillna(0).astype(int)
print(tf)

biner = (tf > 0).astype(int)
print(biner)

print(tf.loc[["pantai", "wisata", "pasir"], "d1"])
print("pantai : wisata =", tf.loc["pantai", "d1"], ":", tf.loc["wisata", "d1"])

for f in [1, 2, 3, 4, 10, 100]:
    print(f"tf = {f:3d} -> 1 + log10(tf) = {1 + math.log10(f):.3f}")

logtf = np.where(tf > 0, 1 + np.log10(tf.clip(lower=1)), 0)
logtf = pd.DataFrame(logtf, index=tf.index, columns=tf.columns)
print(logtf.round(3))

df_t = (tf > 0).sum(axis=1)

tabel_df = logtf.round(3).copy()
tabel_df["df"] = df_t
print(tabel_df)

print("Rare terms (df = 1):", list(df_t[df_t == 1].index))
print("Frequent terms (df >= 3):", list(df_t[df_t >= 3].index))

N = tf.shape[1]
print("N =", N)
print("idf untuk df = 1 :", round(math.log10(N / 1), 3))
print("idf untuk df = 3 :", round(math.log10(N / 3), 3))

idf = np.log10(N / df_t)
tabel_idf = logtf.round(3).copy()
tabel_idf["df"] = df_t
tabel_idf["idf"] = idf.round(3)
print(tabel_idf)

tfidf = logtf.mul(idf, axis=0)
print(tfidf.round(3))

# Pembanding: Raw TF x IDF
tfidf_mentah = tf.mul(idf, axis=0)
print(tfidf_mentah.round(3))

for d in tfidf.columns:
    print(d, tfidf[d].nlargest(3).round(3).to_dict())

banding = pd.DataFrame({
    "tf": tf["d1"],
    "biner": biner["d1"],
    "logtf": logtf["d1"].round(3),
    "df": df_t,
    "idf": idf.round(3),
    "tf-idf": tfidf["d1"].round(3)
})
print(banding.loc[["pantai", "wisata", "pasir", "laut"]])

banding.loc[["pantai", "wisata", "pasir", "laut"],
            ["tf", "logtf", "tf-idf"]].plot(kind="bar")
plt.title("Bobot term pada d1 menurut skema")
plt.ylabel("Bobot")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

cek = {
    ("pantai", "d1"): round((1 + math.log10(3)) * math.log10(5 / 2), 3),
    ("pasir", "d1"): round((1 + math.log10(2)) * math.log10(5 / 1), 3),
    ("wisata", "d1"): round((1 + math.log10(2)) * math.log10(5 / 4), 3)
}

for (t, d), nilai in cek.items():
    hitung = round(tfidf.loc[t, d], 3)
    status = "OK" if abs(hitung - nilai) < 0.001 else "BEDA"
    print(f"{t:8} {d} rumus = {nilai:.3f} kode = {hitung:.3f} {status}")

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

teks = [" ".join(token[d]) for d in tf.columns]

# 1) Biner
Xb = CountVectorizer(binary=True, vocabulary=urutan).fit_transform(teks)
print("Verifikasi biner:", np.array_equal(Xb.toarray().T, biner.values))

# 2) Log TF: scikit-learn memakai logaritma natural
Xl = TfidfVectorizer(vocabulary=urutan, use_idf=False, norm=None,
                     sublinear_tf=True).fit_transform(teks)
logtf_ln = np.where(tf.values > 0, 1 + np.log(tf.values.clip(min=1)), 0)
print("Verifikasi log TF ln:", np.allclose(Xl.toarray().T, logtf_ln))

# 3) TF-IDF bawaan scikit-learn
T = TfidfVectorizer(vocabulary=urutan).fit_transform(teks)
print("Bentuk matriks TF-IDF scikit-learn:", T.shape)
print("Matriks TF-IDF scikit-learn:")
print(pd.DataFrame(T.toarray().T, index=urutan, columns=tf.columns).round(3))