import streamlit as st
import os
import sys

# Ana klasörü sys.path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db, save_chunks, clear_db
from src.ingest import process_document
from src.retriever import retrieve_smart_chunks
from src.llm import LLMEngine
from src.verifier import verify_citations

# 🏛️ Doğal, Klasik Masaüstü Uygulaması Standart Sayfa Ayarları
st.set_page_config(
    page_title="Verifiable Local RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sol paneli genişleten ve alt satıra kırmayan Özel CSS
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            min-width: 340px !important;
            max-width: 400px !important;
        }
        .uploaded-file-item {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 14px;
            padding: 4px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Yerel veritabanı başlatma
init_db()

@st.cache_resource
def get_llm_engine():
    return LLMEngine()

llm_engine = get_llm_engine()

# Session State Yönetimi
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "lang" not in st.session_state:
    st.session_state.lang = "TR"

# --- İÇERİK DİL SÖZLÜĞÜ (i18n) ---
TEXTS = {
    "TR": {
        "title": "Verifiable Local RAG",
        "caption": "Çevrimdışı Doküman Analizi ve Alıntı Doğrulama",
        "foundry_active": "🟢 Foundry Local SDK Aktif (Yerel LLM Modu)",
        "fallback_active": "🟠 Fallback Vektör Motoru Aktif (Deterministik Mod)",
        "doc_upload": "Doküman Yükleme",
        "upload_label": "PDF veya TXT dosyası ekleyin:",
        "processing": "İşleniyor:",
        "uploaded": "Yüklendi:",
        "search_scope": "Arama Kapsamı",
        "select_doc": "Aranacak Belge:",
        "all_docs": "Tüm Belgeler",
        "loaded_docs": "Yüklü Belgeler",
        "no_docs": "Henüz belge yüklenmedi.",
        "reset_btn": "Veritabanını ve Sohbeti Sıfırla",
        "reset_toast": "Veritabanı ve sohbet geçmişi sıfırlandı!",
        "header": "Doğrulanabilir Doküman Asistanı",
        "sub_header": "Yüklenen belgeler üzerinden kaynak alıntılı ve doğrulanmış bilgi sunar.",
        "tab_chat": "Sohbet ve Alıntılar",
        "tab_db": "Veritabanı İnceleyici",
        "input_placeholder": "Sorunuzu yazın...",
        "spinner": "Yanıt ve alıntılar hazırlanıyor...",
        "not_found": "Yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunamadı.",
        "score_label": "Alıntı Doğruluk Skoru",
        "status_label": "Doğrulama Durumu",
        "verified_sources": "Doğrulanan Kaynaklar:",
        "db_title": "SQLite Vektör Mağazası",
        "db_sub": "Veritabanı kayıtlarını inceleyin.",
        "db_sim_label": "Arama simülasyonu için kelime girin:",
        "db_results": "için en alakalı 5 parça:",
        "lang_label": "🌐 Arayüz Dili / Interface Language"
    },
    "EN": {
        "title": "Verifiable Local RAG",
        "caption": "Offline Document Analysis & Citation Verification",
        "foundry_active": "🟢 Foundry Local SDK Active (Local LLM Mode)",
        "fallback_active": "🟠 Fallback Vector Engine Active (Deterministic Mode)",
        "doc_upload": "Document Upload",
        "upload_label": "Add PDF or TXT file:",
        "processing": "Processing:",
        "uploaded": "Uploaded:",
        "search_scope": "Search Scope",
        "select_doc": "Target Document:",
        "all_docs": "All Documents",
        "loaded_docs": "Loaded Documents",
        "no_docs": "No documents uploaded yet.",
        "reset_btn": "Reset Database & Chat",
        "reset_toast": "Database and chat history reset!",
        "header": "Verifiable Document Assistant",
        "sub_header": "Provides verified insights with source citations from uploaded documents.",
        "tab_chat": "Chat & Citations",
        "tab_db": "Database Explorer",
        "input_placeholder": "Ask your question...",
        "spinner": "Generating answer and verifying citations...",
        "not_found": "Insufficient information found in the uploaded documents for this question.",
        "score_label": "Citation Accuracy Score",
        "status_label": "Verification Status",
        "verified_sources": "Verified Sources:",
        "db_title": "SQLite Vector Store",
        "db_sub": "Inspect local vector database records.",
        "db_sim_label": "Enter text for search simulation:",
        "db_results": "top 5 most relevant chunks for:",
        "lang_label": "🌐 Interface Language / Arayüz Dili"
    }
}

# --- SOL PANEL (Sidebar) ---
with st.sidebar:
    st.title(TEXTS[st.session_state.lang]["title"])
    st.caption(TEXTS[st.session_state.lang]["caption"])
    
    # Language Toggle Switch
    selected_lang = st.radio(
        TEXTS[st.session_state.lang]["lang_label"],
        options=["TR", "EN"],
        horizontal=True,
        index=0 if st.session_state.lang == "TR" else 1
    )
    if selected_lang != st.session_state.lang:
        st.session_state.lang = selected_lang
        st.rerun()
        
    st.divider()
    
    # Active Model Engine Status Indicator
    if getattr(llm_engine, "is_foundry_active", False):
        st.success(TEXTS[st.session_state.lang]["foundry_active"])
    else:
        st.warning(TEXTS[st.session_state.lang]["fallback_active"], icon="⚠️")
        
    st.divider()
    
    st.subheader(TEXTS[st.session_state.lang]["doc_upload"])
    uploaded_files_batch = st.file_uploader(
        TEXTS[st.session_state.lang]["upload_label"], 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files_batch:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for file in uploaded_files_batch:
            file_path = os.path.join(data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"{TEXTS[st.session_state.lang]['processing']} {file.name}..."):
                    chunks = process_document(file_path)
                    for chunk in chunks:
                        chunk["embedding"] = llm_engine.generate_embedding(chunk["content"])
                    save_chunks(chunks)
                    st.session_state.uploaded_files.append(file.name)
                    st.toast(f"{TEXTS[st.session_state.lang]['uploaded']} {file.name}")
                    
    st.divider()
    st.subheader(TEXTS[st.session_state.lang]["search_scope"])
    all_docs_label = TEXTS[st.session_state.lang]["all_docs"]
    doc_options = [all_docs_label] + st.session_state.uploaded_files
    selected_doc_filter = st.selectbox(
        TEXTS[st.session_state.lang]["select_doc"],
        options=doc_options,
        index=0
    )
    
    st.divider()
    st.subheader(TEXTS[st.session_state.lang]["loaded_docs"])
    if st.session_state.uploaded_files:
        for idx, f in enumerate(st.session_state.uploaded_files, 1):
            st.markdown(f'<div class="uploaded-file-item" title="{f}">📄 <b>{idx}.</b> {f}</div>', unsafe_allow_html=True)
    else:
        st.caption(TEXTS[st.session_state.lang]["no_docs"])
        
    st.divider()
    if st.button(TEXTS[st.session_state.lang]["reset_btn"], use_container_width=True):
        clear_db()
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.toast(TEXTS[st.session_state.lang]["reset_toast"])
        st.rerun()

# --- ANA EKRAN ---
st.header(TEXTS[st.session_state.lang]["header"])
st.write(TEXTS[st.session_state.lang]["sub_header"])

tab1, tab2 = st.tabs([TEXTS[st.session_state.lang]["tab_chat"], TEXTS[st.session_state.lang]["tab_db"]])

with tab1:
    # 1. Önce Sohbet Geçmişini Render Et
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "verification" in msg and msg["verification"]:
                v = msg["verification"]
                st.divider()
                st.caption(f"{TEXTS[st.session_state.lang]['score_label']}: %{v['confidence_score']} | {TEXTS[st.session_state.lang]['status_label']}: {v['verification_status']}")
                if v["verified_citations"]:
                    st.caption(f"{TEXTS[st.session_state.lang]['verified_sources']} " + ", ".join([f"{c}" for c in v["verified_citations"]]))

    # 2. Soru Girişi Kutusunu En Alta Sabitle
    if prompt := st.chat_input(TEXTS[st.session_state.lang]["input_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner(TEXTS[st.session_state.lang]["spinner"]):
                query_vec = llm_engine.generate_embedding(prompt)
                
                retrieved_chunks = retrieve_smart_chunks(
                    query_text=prompt,
                    query_embedding=query_vec,
                    top_k=3,
                    filter_source=selected_doc_filter
                )
                
                response_text = llm_engine.generate_answer(prompt, retrieved_chunks)
                verification = verify_citations(response_text, retrieved_chunks)
                
                # Çirkin || boru sembollerini temizle
                cleaned_disp = response_text.replace("|||---|---|---|---|", "").replace("|||", "").replace("||", "")
                
                if verification["confidence_score"] == 0.0 or "bulunmamaktadır" in response_text.lower():
                    st.warning(TEXTS[st.session_state.lang]["not_found"])
                else:
                    st.markdown(cleaned_disp)
                
                # Panel
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(TEXTS[st.session_state.lang]["score_label"], f"%{verification['confidence_score']}")
                with col2:
                    st.metric(TEXTS[st.session_state.lang]["status_label"], verification['verification_status'])
                    
                if verification["verified_citations"]:
                    st.write(f"**{TEXTS[st.session_state.lang]['verified_sources']}**")
                    for cit in verification["verified_citations"]:
                        st.info(f"Kaynak: {cit}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": TEXTS[st.session_state.lang]["not_found"] if verification["confidence_score"] == 0.0 else cleaned_disp,
            "verification": verification
        })

with tab2:
    st.subheader(TEXTS[st.session_state.lang]["db_title"])
    st.write(TEXTS[st.session_state.lang]["db_sub"])
    
    debug_query = st.text_input(TEXTS[st.session_state.lang]["db_sim_label"])
    if debug_query:
        q_vec = llm_engine.generate_embedding(debug_query)
        results = retrieve_smart_chunks(
            query_text=debug_query,
            query_embedding=q_vec,
            top_k=5,
            filter_source=selected_doc_filter
        )
        
        st.write(f"'{debug_query}' {TEXTS[st.session_state.lang]['db_results']}")
        for res in results:
            with st.expander(f"{res['source_file']} (Sayfa {res['page_number']}) - Skor: %{round(res['similarity_score']*100, 2)}"):
                st.write(res["content"])
                st.caption(f"Chunk Index: {res['chunk_index']} | DB ID: {res['id']}")
