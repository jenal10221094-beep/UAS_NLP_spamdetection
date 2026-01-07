# 📧 Sistem Deteksi Spam SMS (UAS Natural Language Processing)

Proyek ini adalah sistem klasifikasi teks otomatis yang dirancang untuk mendeteksi apakah sebuah pesan termasuk kategori **Spam** (pesan sampah/penipuan) atau **Ham** (pesan normal). Sistem ini dibangun untuk memenuhi tugas UAS dengan mengikuti alur *NLP Project Life Cycle*.

## 🛠️ Alur Sistem (Pipeline)
Sistem ini bekerja secara sistematis mengikuti tahapan berikut sesuai dengan standar pemrosesan bahasa alami:

1.  **Data Collection**: Mengambil dataset dari `data/spam_dataset.csv`. Dataset terdiri dari label (`v1`) dan teks pesan (`v2`).
2.  **Text Preprocessing**: Proses pembersihan teks yang meliputi:
    * **Lowercasing**: Mengubah teks menjadi huruf kecil.
    * **Normalizing**: Menghapus angka dan karakter khusus menggunakan RegEx.
    * **Tokenizing**: Memecah kalimat menjadi potongan kata.
    * **Stopword Removal**: Menghapus kata umum yang tidak memiliki makna penting (seperti 'is', 'the', dll).
    * **Stemming**: Mengubah kata ke bentuk dasar menggunakan PorterStemmer.
3.  **Feature Engineering**: Menggunakan **TF-IDF Vectorizer** untuk mengubah teks bersih menjadi vektor angka berbasis bobot kepentingan kata.
4.  **Modeling**: Pelatihan menggunakan dua algoritma utama:
    * **Naive Bayes** (MultinomialNB)
    * **Support Vector Machine** (SVM)
5.  **Evaluation**: Performa diukur menggunakan metrik **Accuracy, Precision, Recall, dan F1-Score** melalui *Classification Report*.
6.  **Deployment**: Implementasi model ke dalam dashboard interaktif menggunakan **Streamlit**.



## 📁 Struktur Folder
```text
uas_NLP_spam/
├── data/
│   └── spam_dataset.csv       # Dataset SMS Spam (v1, v2)
├── models/
│   ├── model_nb.pkl           # Hasil export model Naive Bayes
│   ├── model_svm.pkl          # Hasil export model SVM
│   └── tfidf_vectorizer.pkl   # Hasil export TF-IDF
├── src/
│   ├── preprocessing.py       # Logika pembersihan teks
│   └── train.py               # Script utama pelatihan model
├── README.md                  # Dokumentasi proyek
└── app.py                     # Dashboard Streamlit (Interface)

🚀 Cara Menjalankan Sistem
1. Persiapan Environment
Pastikan Anda berada di environment conda yang benar:

Bash

conda activate uas_NLP_spam
pip install pandas scikit-learn nltk streamlit

2. Pelatihan Model (Training)
Jalankan script berikut untuk memproses data dan menghasilkan file model .pkl:

Bash

python src/train.py

3. Menjalankan Dashboard (Deployment)
Gunakan Streamlit untuk membuka antarmuka prediksi:

Bash

streamlit run app.py

📊 Contoh Hasil Evaluasi
Setelah menjalankan train.py, sistem akan menampilkan laporan evaluasi di terminal yang mencakup:

Precision: Ketepatan dalam memprediksi spam.

Recall: Kemampuan menangkap seluruh pesan spam.

F1-Score: Keseimbangan antara precision dan recall.

Accuracy: Persentase kebenaran prediksi secara keseluruhan.

Dibuat Oleh: [Nama Anda] Mata Kuliah: Natural Language Processing (NLP) Kampus: STT Cipasung


### Tips Penggunaan di VS Code:
1. Klik kanan di folder utama proyek Anda (`uas_NLP_spam`).
2. Pilih **New File** dan beri nama `README.md`.
3. Tempelkan kode di atas ke dalam file tersebut.
4. Tekan `Ctrl + S` untuk menyimpan.
5. Anda bisa melihat tampilannya dengan klik tombol **"Open Preview"** (ikon kaca pembesar di pojok kanan atas VS Code).