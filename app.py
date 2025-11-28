import streamlit as st
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import google.generativeai as genai
from datetime import datetime
import io

# Page configuration
st.set_page_config(
    page_title="Market Research RAG Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .analysis-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #fff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Functions
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text_list, chunk_size=800):
    chunks = []
    for page_num, page_text in enumerate(text_list):
        words = page_text.split()
        for i in range(0, len(words), chunk_size//4):
            chunk = ' '.join(words[i:i+chunk_size//4])
            if len(chunk) > 50:
                chunks.append({
                    'text': chunk,
                    'page': page_num + 1
                })
    return chunks

def process_pdf(pdf_file, api_key):
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Read PDF
        pdf_bytes = pdf_file.read()
        pdf_file_obj = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_file_obj)
        
        pages = [page.extract_text() for page in reader.pages]
        
        # Chunk text
        chunks = chunk_text(pages)
        
        # Load embedding model
        model = load_embedding_model()
        
        # Generate embeddings
        chunk_texts = [c['text'] for c in chunks]
        embeddings = model.encode(chunk_texts).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return chunks, index, model, len(reader.pages)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None, None, None, 0

def analyze_question(question, chunks, index, model, api_key, num_chunks=5):
    try:
        # Search relevant chunks
        q_emb = model.encode([question]).astype('float32')
        distances, indices = index.search(q_emb, num_chunks)
        
        # Build context
        context_parts = []
        source_pages = []
        for i, idx in enumerate(indices[0]):
            chunk = chunks[idx]
            context_parts.append(f"[Excerpt {i+1} - Page {chunk['page']}]:\n{chunk['text'][:500]}...")
            source_pages.append(chunk['page'])
        
        context = "\n\n".join(context_parts)
        
        # Generate response with Gemini
        model_gemini = genai.GenerativeModel('models/gemini-2.0-flash-exp')
        
        prompt = f"""You are a marketing professor with 20+ years of experience analyzing academic papers.

From this research paper context ONLY:

📄 PAPER EXCERPTS:
{context}

🔍 RESEARCH QUESTION: {question}

Please provide:
1. **DIRECT ANSWER** (2-3 sentences)
2. **KEY EVIDENCE** from paper (quote relevant parts with page references)
3. **ACADEMIC INSIGHT** (what this means for marketing theory/practice)

Keep it clear, structured, and insightful."""

        response = model_gemini.generate_content(prompt)
        
        return {
            'question': question,
            'answer': response.text,
            'sources': list(set(source_pages)),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'relevance_scores': distances[0].tolist()
        }
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None

# Header
st.markdown('<p class="main-header">📊 Market Research RAG Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Academic Paper Analysis using Retrieval-Augmented Generation</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Google Gemini API Key", type="password", help="Enter your Google Gemini API key")
    
    st.markdown("---")
    
    st.header("📤 Upload Document")
    uploaded_file = st.file_uploader("Upload Marketing Research PDF", type=['pdf'])
    
    chunk_size = st.slider("Chunk Size (words)", 400, 1200, 800, 100)
    num_retrieval_chunks = st.slider("Retrieval Chunks", 3, 10, 5)
    
    if uploaded_file and api_key:
        if st.button("🚀 Process PDF"):
            with st.spinner("Processing PDF... This may take a moment."):
                chunks, index, model, num_pages = process_pdf(uploaded_file, api_key)
                if chunks and index and model:
                    st.session_state.chunks = chunks
                    st.session_state.index = index
                    st.session_state.model = model
                    st.session_state.processed = True
                    st.session_state.num_pages = num_pages
                    st.success(f"✅ PDF processed successfully! ({len(chunks)} chunks from {num_pages} pages)")
    
    if st.session_state.processed:
        st.markdown("---")
        st.header("📈 Document Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.num_pages)
        with col2:
            st.metric("Chunks", len(st.session_state.chunks))
        
        st.markdown("---")
        if st.button("🔄 Reset Analysis"):
            st.session_state.processed = False
            st.session_state.chunks = []
            st.session_state.index = None
            st.session_state.model = None
            st.session_state.analysis_history = []
            st.rerun()

# Main content
if not st.session_state.processed:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📄</h3>
            <h4>Upload PDF</h4>
            <p>Upload your marketing research paper</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍</h3>
            <h4>AI Analysis</h4>
            <p>Ask questions about the research</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>💡</h3>
            <h4>Get Insights</h4>
            <p>Receive expert-level analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🎯 How It Works")
    st.markdown("""
    1. **Upload** your marketing research paper (PDF format)
    2. **Enter** your Gemini API key in the sidebar
    3. **Click** "Process PDF" to analyze the document
    4. **Ask** questions about marketing concepts, frameworks, or findings
    5. **Receive** academically-rigorous answers with evidence from the paper
    """)
    
    st.header("💡 Sample Questions You Can Ask")
    st.markdown("""
    - How does the 20th-century marketing mix differ from the 21st-century version?
    - What new marketing mix components are proposed by the authors?
    - What are the paper's main conclusions about marketing theory?
    - What empirical evidence supports the proposed framework?
    - How do the authors critique traditional marketing approaches?
    """)

else:
    # Analysis interface
    tab1, tab2, tab3 = st.tabs(["🔍 Ask Questions", "📚 Quick Analysis", "📜 History"])
    
    with tab1:
        st.header("Ask Your Research Question")
        
        question = st.text_area(
            "Enter your question about the research paper:",
            height=100,
            placeholder="e.g., What are the key differences between traditional and modern marketing approaches according to this paper?"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_btn = st.button("🔍 Analyze Question", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        
        if analyze_btn and question:
            with st.spinner("Analyzing your question..."):
                result = analyze_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.index,
                    st.session_state.model,
                    api_key,
                    num_retrieval_chunks
                )
                
                if result:
                    st.session_state.analysis_history.append(result)
                    
                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                    st.markdown(f"### 📝 Question\n{result['question']}")
                    st.markdown("---")
                    st.markdown(f"### 💡 Analysis\n{result['answer']}")
                    st.markdown("---")
                    st.markdown(f"**📍 Source Pages:** {', '.join(map(str, sorted(result['sources'])))}")
                    st.markdown(f"**⏰ Generated:** {result['timestamp']}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.header("Quick Analysis Templates")
        
        preset_questions = [
            "How does the 20th-century marketing mix differ from the 21st-century version according to the paper?",
            "List and explain the updated marketing mix components proposed by the authors.",
            "What are the major findings or conclusions of the paper?",
            "What empirical evidence or case studies are presented in the research?",
            "How do the authors critique traditional marketing frameworks?"
        ]
        
        st.markdown("Click any question to analyze:")
        
        for i, q in enumerate(preset_questions):
            if st.button(f"📌 {q}", key=f"preset_{i}"):
                with st.spinner("Analyzing..."):
                    result = analyze_question(
                        q,
                        st.session_state.chunks,
                        st.session_state.index,
                        st.session_state.model,
                        api_key,
                        num_retrieval_chunks
                    )
                    
                    if result:
                        st.session_state.analysis_history.append(result)
                        
                        st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                        st.markdown(f"### 📝 Question\n{result['question']}")
                        st.markdown("---")
                        st.markdown(f"### 💡 Analysis\n{result['answer']}")
                        st.markdown("---")
                        st.markdown(f"**📍 Source Pages:** {', '.join(map(str, sorted(result['sources'])))}")
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("Analysis History")
        
        if st.session_state.analysis_history:
            for i, result in enumerate(reversed(st.session_state.analysis_history)):
                with st.expander(f"#{len(st.session_state.analysis_history)-i}: {result['question'][:100]}...", expanded=False):
                    st.markdown(f"**Question:** {result['question']}")
                    st.markdown("---")
                    st.markdown(result['answer'])
                    st.markdown("---")
                    st.markdown(f"**Sources:** Pages {', '.join(map(str, sorted(result['sources'])))}")
                    st.markdown(f"**Time:** {result['timestamp']}")
        else:
            st.info("No analysis history yet. Start asking questions!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Powered by Google Gemini 2.0, FAISS & Sentence Transformers</p>
    <p>Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
