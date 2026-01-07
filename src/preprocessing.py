import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Tambahkan baris ini untuk mendownload paket yang error tadi
nltk.download('punkt')
nltk.download('punkt_tab') # <-- BARIS BARU INI WAJIB ADA
nltk.download('stopwords')

def clean_text(text):
    # Pastikan input adalah string untuk menghindari error apply
    text = str(text)
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Normalizing (Menghapus karakter selain huruf a-z)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenizing
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in tokens if w not in stop_words]
    
    # 5. Stemming (Sesuai instruksi Anda)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(w) for w in tokens]
    
    return " ".join(tokens)