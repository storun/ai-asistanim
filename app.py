import streamlit as st
import os
from dotenv import load_dotenv
import tempfile

# --- GEREKLİ KÜTÜPHANELER ---
# Sohbet (Chat) için Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# PDF'i Sayısallaştırmak (Embedding) için Ücretsiz Yerel Model (Kota dostu)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vektör Veritabanı ve PDF Yükleyici
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Ortam Değişkenlerini Yükle (.env dosyasını okur)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# API Anahtarı Kontrolü
if not api_key:
    st.error("HATA: Google API Anahtarı bulunamadı! Lütfen .env dosyanızı kontrol edin.")
    st.stop()

# 2. Sayfa Ayarları
st.set_page_config(page_title="PDF Asistanı", page_icon="📚")
st.title("📚 PDF Kitabınla Sohbet Et")

# 3. Yan Menü: Dosya Yükleme
st.sidebar.header("Döküman Yükle")
uploaded_file = st.sidebar.file_uploader("Bir PDF dosyası yükleyin", type="pdf")

# 4. Ana İşlem Fonksiyonu
def pdf_isleme(file):
    # Geçici dosya oluştur (PyPDFLoader diskten okuma yapar)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.read())
        tmp_path = tmp_file.name

    # PDF'i Yükle ve Parçala
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    
    # Metni küçük parçalara böl (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # --- KRİTİK NOKTA: Yerel Embedding Modeli ---
    # Google yerine bilgisayarın işlemcisini kullanır. Kota harcamaz.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vektör Veritabanı Oluştur
    db = Chroma.from_documents(texts, embeddings)
    
    # Geçici dosyayı temizle
    os.remove(tmp_path)
    
    return db

# 5. Dosya Yüklendiyse İşlemleri Başlat
if uploaded_file is not None:
    with st.spinner("PDF analiz ediliyor, lütfen bekleyin... (İlk seferde model indirilebilir)"):
        try:
            # Veritabanını oluştur
            db = pdf_isleme(uploaded_file)
            st.success("PDF başarıyla analiz edildi! Sorularınızı sorabilirsiniz.")
            
            # --- SOHBET MODELİ: Google Gemini ---
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite-preview-09-2025",
                temperature=0.3, 
                google_api_key=api_key
            )
            
            # Soru-Cevap Zincirini Kur
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever()
            )
            
            # 6. Kullanıcıdan Soru Al
            query = st.text_input("Kitapla ilgili sorunu yaz:")
            
            if query:
                with st.spinner("Cevap hazırlanıyor..."):
                    response = qa_chain.invoke(query)
                    st.write("### 🤖 Cevap:")
                    st.write(response["result"])
                    
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

else:
    st.info("Lütfen sol menüden bir PDF dosyası yükleyin.")