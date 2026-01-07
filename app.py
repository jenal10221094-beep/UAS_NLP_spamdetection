import streamlit as st
import pickle
from src.preprocessing import clean_text

# Load model dan vectorizer
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
model = pickle.load(open('models/model_nb.pkl', 'rb')) # Default menggunakan Naive Bayes

st.title("📧 Spam Detection System")
st.write("Aplikasi pendeteksi pesan spam menggunakan NLP (Naive Bayes)")

input_text = st.text_area("Masukkan pesan yang ingin dicek:")

if st.button("Predict"):
    if input_text:
        # 1. Preprocess
        transformed_text = clean_text(input_text)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_text])
        # 3. Predict
        result = model.predict(vector_input)[0]
        
        # 4. Output
        if result == 'spam' or result == 1:
            st.error("🚨 Hati-hati! Ini adalah SPAM.")
        else:
            st.success("✅ Aman. Ini adalah pesan HAM (Bukan Spam).")
    else:
        st.warning("Silakan masukkan teks terlebih dahulu.")