import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import clean_text

# Membuat folder models jika belum ada
if not os.path.exists('models'):
    os.makedirs('models')

# 1. DATA COLLECTION (Sesuai Gambar 1)
# Menggunakan encoding latin-1 karena file Anda terdeteksi bukan UTF-8
df = pd.read_csv('data/spam_dataset.csv', encoding='latin-1')

# 2. PREPROCESSING & CLEANING
# Mengambil hanya kolom v1 (label) dan v2 (text), lalu mengganti namanya
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Menghapus data kosong jika ada
df.dropna(inplace=True)

print("Tahap Preprocessing sedang berjalan...")
# Menjalankan lowercasing, tokenizing, stopword removal, stemming (Sesuai tugas Anda)
df['clean_text'] = df['text'].apply(clean_text)

# 3. FEATURE ENGINEERING (Sesuai Gambar 1: TF-IDF)
print("Ekstraksi Fitur dengan TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label']

# Split data: 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODELING (Sesuai Gambar 2: NB & SVM)
models = {
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True)
}

# 5. EVALUATION (Sesuai Gambar 2: Accuracy, Precision, Recall, F1-Score)
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n================ Hasil Evaluasi {name} ================")
    # Menampilkan Precision, Recall, F1-Score sesuai permintaan tugas
    print(classification_report(y_test, y_pred)) 
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

# 6. OUTPUT SISTEM (Penyimpanan Model .pkl)
print("\nMenyimpan hasil model...")
pickle.dump(tfidf, open('models/tfidf_vectorizer.pkl', 'wb'))
pickle.dump(models["Naive Bayes"], open('models/model_nb.pkl', 'wb'))
pickle.dump(models["SVM"], open('models/model_svm.pkl', 'wb'))

print("Proses Selesai! File .pkl telah dibuat di folder 'models'.")