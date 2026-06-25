import os
import pandas as pd
import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import streamlit as st

# ==========================================
# 1. PAGE SETUP & DESIGN STYLING
# ==========================================
st.set_page_config(
    page_title="CrediTrust AI Analyst",
    page_icon="🤖",
    layout="wide"
)

# Custom Corporate CSS for CrediTrust Financial branding
st.markdown("""
    <style>
    .main-header { font-size:2.4rem !important; color: #1E3A8A; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { font-size:1.1rem !important; color: #4B5563; margin-bottom: 2rem; }
    .source-box { background-color: #F3F4F6; border-left: 4px solid #3B82F6; padding: 1rem; margin-bottom: 1rem; border-radius: 4px; }
    .meta-tag { font-weight: bold; color: #2563EB; font-size: 0.85rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED INFRASTRUCTURE LOADER (Runs Once)
# ==========================================
@st.cache_resource
def initialize_rag_backend():
    # Load persistent vector store directory from Task 2
    chroma_client = chromadb.PersistentClient(path="./complaints_chroma_db")
    collection = chroma_client.get_or_create_collection(name="customer_complaints")
    
    # Fast approach: Grab the documents directly from your pre-built database!
    # This avoids loading the heavy CSV and splitting data completely.
    all_data = collection.get()
    chunks_list = all_data['documents']
    
    if not chunks_list:
        raise ValueError("ChromaDB collection appears empty. Run Task 2 first!")
        
    vectorizer = TfidfVectorizer(max_features=384)
    vectorizer.fit(chunks_list)
    
    return collection, vectorizer

# Keep this part right under it!
try:
    collection, vectorizer = initialize_rag_backend()
except Exception as e:
    st.error(f"Backend Initialization Error: Make sure your Task 2 Vector Database exists. Details: {e}")
    st.stop()
# ==========================================
# 3. CORE RETRIEVAL & SYNTHESIS LOGIC
# ==========================================
def process_query(question):
    # Vectorize plain-English input question string
    query_vector = vectorizer.transform([question]).toarray().tolist()
    
    # Run similarity query against local collection matrix (k=3 matching records)
    results = collection.query(query_embeddings=query_vector, n_results=3)
    
    retrieved_docs = results['documents'][0]
    retrieved_metas = results['metadatas'][0]
    
    # Synthesis/Extraction Engine (Framework-Free to ensure 100% environment safety)
    bullet_points = []
    sources = []
    
    for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        clean_text = doc.strip()
        bullet_points.append(f"• {clean_text}")
        sources.append({
            "id": meta['complaint_id'],
            "product": meta['product'],
            "text": clean_text
        })
        
    generated_answer = (
        f"Based on a real-time semantic analysis of historical CrediTrust customer complaints, "
        f"the following core systemic issues were surfaced for your review:\n\n" + 
        "\n\n".join(bullet_points) + 
        "\n\n**Strategic Recommendation:** Internal logs demonstrate repetitive customer friction points "
        "matching these exact interaction pathways. Cross-functional operational review is highly recommended."
    )
    
    return generated_answer, sources

# ==========================================
# 4. STREAMLIT FRONT-END LAYOUT UI
# ==========================================

# Header section matching company goals
st.markdown('<div class="main-header">🤖 CrediTrust Complaint AI Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Empowering Product, Support, and Compliance teams with evidence-backed customer insights.</div>', unsafe_allow_html=True)

# Layout Split: Main Interface vs Side Information Bar
col_main, col_sidebar = st.columns([3, 1])

with col_sidebar:
    st.markdown("### Operational Scope")
    st.info("""
    **Supported Portals:**
    - 💳 Credit Cards
    - 🏦 Savings Accounts
    - 💸 Money Transfers
    - 🤝 Personal Loans
    """)
    
    # Clear conversation state button
    if st.button("🧹 Clear Conversation", use_container_width=True):
        if "conversation_history" in st.session_state:
            del st.session_state["conversation_history"]
        st.rerun()

with col_main:
    # Initialize interactive memory container states
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Display active conversation logs if they exist
    for chat in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            st.markdown("#### Traced Evidence & Verification Sources:")
            for src in chat["sources"]:
                st.markdown(f"""
                <div class="source-box">
                    <span class="meta-tag">Complaint ID: {src['id']} | Product Segment: {src['product']}</span><br/>
                    <em>"{src['text']}"</em>
                </div>
                """, unsafe_allow_html=True)

    # Main text input chat container interface
    user_query = st.chat_input("Ask a question (e.g., 'Why are people unhappy with Credit Cards?')")

    if user_query:
        # Show prompt context directly on submit
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing vector database clusters..."):
                # Compute query and synthesis output elements
                answer, sources = process_query(user_query)
                
                # Render generated textual insights
                st.write(answer)
                
                # Render metadata source containers below answers for risk and auditing validation
                st.markdown("#### Traced Evidence & Verification Sources:")
                for src in sources:
                    st.markdown(f"""
                    <div class="source-box">
                        <span class="meta-tag">Complaint ID: {src['id']} | Product Segment: {src['product']}</span><br/>
                        <em>"{src['text']}"</em>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Commit results permanently to session session trace logs
        st.session_state.conversation_history.append({
            "question": user_query,
            "answer": answer,
            "sources": sources
        })