import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- BAŞLANGIÇ AYARLARI VE SQLITE YAMASI ---
# Streamlit Cloud'da ChromaDB hatasını önlemek için bu kısım şarttır.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -------------------------------------------

# --- KÜTÜPHANELER ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Import hatasını önlemek için daha güvenli bir yol:
from langchain.chains import RetrievalQA

# 1. API Anahtarını Yükle (Environment veya Secrets'tan)
load_dotenv()
# Streamlit Secrets kontrolü (Bulutta çalışırken burası devreye girer)
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    api_key = os.getenv("GOOGLE_API_KEY")

# 2. Sayfa Ayarları
st.set_page_config(page_title="PDF Asistanı", page_icon="🤖")
st.title("📄 PDF Dosyanla Sohbet Et")

# 3. Embedding Modeli (Yerel & Ücretsiz - Kota Dostu)
@st.cache_resource # Modeli önbelleğe alarak hızlandırır
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Dosya İşleme Fonksiyonu
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = get_embedding_model()
    # ChromaDB'yi geçici bellekte çalıştır
    db = Chroma.from_documents(texts, embeddings)
    
    os.remove(tmp_path)
    return db

# 5. Arayüz ve Akış
st.sidebar.header("Döküman Yükle")
uploaded_file = st.sidebar.file_uploader("PDF Seç", type="pdf")

if uploaded_file:
    with st.spinner("PDF analiz ediliyor... (İlk seferde model inebilir)"):
        try:
            db = process_pdf(uploaded_file)
            st.success("Analiz tamamlandı! Sorunu sorabilirsin.")

            # --- SOHBET MODELİ: Google Gemini ---
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                temperature=0.3, 
                google_api_key=api_key
            )

            # Soru-Cevap Zinciri
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever()
            )

            user_q = st.text_input("Soru:")
            if user_q:
                resp = qa.invoke(user_q)
                st.write("### 🤖 Cevap:")
                st.write(resp["result"])

        except Exception as e:
            st.error(f"Hata oluştu: {e}")
else:
    st.info("Lütfen başlamak için sol menüden bir PDF yükleyin.")
