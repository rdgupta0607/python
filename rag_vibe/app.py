import streamlit as st
import os
import shutil
import time
import datetime
import pypdf
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import numpy as np
import re
import uuid
from typing import List, Dict, Any, Tuple

# --- Configuration ---
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHROMA_DB_PATH = "./chroma_db"
LOCAL_MODEL_PATH = "./all-MiniLM-L12-v2"  # User specified folder
FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# --- Helper Classes & Functions ---

def get_page_for_index(index: int, page_boundaries: List[Tuple[int, int, int]]) -> int:
    """Returns page number for a given character index."""
    for page_num, start, end in page_boundaries:
        if start <= index < end:
            return page_num
    return page_boundaries[-1][0] if page_boundaries else 1

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """Splits text into chunks, returns list of (text_chunk, start_idx, end_idx)."""
        final_chunks = []
        if not text:
            return []
        
        # Simple implementation of recursive splitting
        self._split_recursive(text, 0, final_chunks)
        return final_chunks

    def _split_recursive(self, text: str, current_offset: int, chunks: List[Tuple[str, int, int]]):
        if len(text) <= self.chunk_size:
            chunks.append((text, current_offset, current_offset + len(text)))
            return

        # Find best separator
        split_idx = -1
        separator_used = ""
        for sep in self.separators:
            if sep == "":
                split_idx = self.chunk_size
                separator_used = ""
                break
            
            # Find last occurrence of separator within limit
            match = text[:self.chunk_size].rfind(sep)
            if match != -1:
                split_idx = match + len(sep)
                separator_used = sep
                break
        
        if split_idx == -1:
            split_idx = self.chunk_size

        chunk_text = text[:split_idx]
        chunks.append((chunk_text, current_offset, current_offset + len(chunk_text)))
        
        # Prepare next chunk with overlap
        next_start_in_text = max(0, split_idx - self.chunk_overlap)
        
        # Safety check to avoid infinite loops
        if next_start_in_text >= split_idx:
             next_start_in_text = max(0, split_idx - 1) 
        
        remaining_text = text[next_start_in_text:]
        
        if not remaining_text or len(remaining_text) == len(text):
             return

        self._split_recursive(remaining_text, current_offset + next_start_in_text, chunks)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loads the sentence transformer model."""
    try:
        # Try local folder first as requested
        if os.path.exists(LOCAL_MODEL_PATH):
            return SentenceTransformer(LOCAL_MODEL_PATH)
        # Fallback to downloading/cache
        return SentenceTransformer(FALLBACK_MODEL_NAME)
    except Exception as e:
        # If local fails, hard fallback
        return SentenceTransformer(FALLBACK_MODEL_NAME)

def get_chroma_client():
    """Returns a persistent ChromaDB client."""
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name="pdf_documents",
        metadata={"hnsw:space": "cosine"}
    )

def clear_collection():
    client = get_chroma_client()
    try:
        client.delete_collection("pdf_documents")
    except:
        pass
    get_collection() # Recreate

def extract_text_from_pdf(file_stream) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Extracts text and returns (full_text, page_boundaries).
    page_boundaries is list of (page_num, start_char_idx, end_char_idx)
    """
    reader = pypdf.PdfReader(file_stream)
    full_text = ""
    page_boundaries = []
    
    current_idx = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_len = len(text)
        page_num = i + 1
        page_boundaries.append((page_num, current_idx, current_idx + text_len))
        full_text += text
        current_idx += text_len
        
    return full_text, page_boundaries

def process_and_index_documents(uploaded_files, chunk_size, chunk_overlap):
    model = load_embedding_model()
    collection = get_collection()
    
    total_chunks = 0
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {file.name}...")
            # We need to handle file pointer for multiple reads if necessary, or just read once.
            # pypdf reads from stream.
            full_text, boundaries = extract_text_from_pdf(file)
            
            splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
            raw_chunks = splitter.split_text(full_text)
            
            if not raw_chunks:
                results.append({"name": file.name, "status": "No text extracted", "chunks": 0, "pages": len(boundaries)})
                continue

            # Prepare data for Chroma
            ids = []
            documents = []
            metadatas = []
            
            timestamp = datetime.datetime.now().isoformat()
            
            chunk_texts = [rc[0] for rc in raw_chunks]
            # Batch encode
            chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
            
            for i, (chunk_text, start, end) in enumerate(raw_chunks):
                if not chunk_text.strip():
                    continue
                    
                start_page = get_page_for_index(start, boundaries)
                end_page = get_page_for_index(end - 1, boundaries)
                page_str = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
                
                chunk_id = f"{file.name}_{uuid.uuid4().hex}"
                
                ids.append(chunk_id)
                documents.append(chunk_text)
                metadatas.append({
                    "source_file": file.name,
                    "page_number": page_str,
                    "chunk_index": i,
                    "upload_timestamp": str(timestamp)
                })
            
            if ids:
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=chunk_embeddings.tolist(),
                    metadatas=metadatas
                )
                
            total_chunks += len(ids)
            results.append({"name": file.name, "status": "Success", "chunks": len(ids), "pages": len(boundaries)})
            
        except Exception as e:
            results.append({"name": file.name, "status": f"Failed: {str(e)}", "chunks": 0, "pages": 0})
            
        progress_bar.progress((idx + 1) / len(uploaded_files))
        
    status_text.text("Processing complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return results

def hybrid_retrieval(query, k_retrieval, k_rerank, api_key):
    """
    1. Semantic Search (Chroma)
    2. Lexical Reranking (Keyword Overlap)
    """
    model = load_embedding_model()
    collection = get_collection()
    
    # Semantic Search
    query_embedding = model.encode([query], normalize_embeddings=True).tolist()
    
    # Fetch larger candidate set
    fetch_k = max(k_retrieval * 4, 60) 
    
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        return []

    if not results['documents'] or not results['documents'][0]:
        return []
    
    # Flatten results
    candidates = []
    ids = results['ids'][0]
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    dists = results['distances'][0]
    
    for i in range(len(ids)):
        similarity = 1 - dists[i] # Cosine distance to similarity
        candidates.append({
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "semantic_score": similarity
        })
        
    # Lexical Reranking
    query_tokens = set(re.findall(r'\w+', query.lower()))
    
    for cand in candidates:
        doc_tokens = set(re.findall(r'\w+', cand['text'].lower()))
        if not query_tokens:
            overlap_score = 0
        else:
            overlap = len(query_tokens.intersection(doc_tokens))
            overlap_score = overlap / len(query_tokens)
        
        cand['lexical_score'] = overlap_score
        
    # Hybrid Score (0.7 semantic, 0.3 lexical)
    for cand in candidates:
        cand['hybrid_score'] = (cand['semantic_score'] * 0.7) + (cand['lexical_score'] * 0.3)
        
    # Sort
    candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    # Filter
    final_candidates = candidates[:12] # Max 12
    return final_candidates

def generate_rag_response(query, relevant_chunks, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    context_text = ""
    for idx, chunk in enumerate(relevant_chunks):
        source = chunk['metadata'].get('source_file', 'Unknown')
        page = chunk['metadata'].get('page_number', '?')
        text = chunk['text']
        context_text += f"[Source: {source}, Page: {page}]\n{text}\n\n"
        
    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENTS:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite your sources by mentioning the document name and page number
- Be concise but comprehensive
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --- Main App Logic ---

# Sidebar
with st.sidebar:
    st.title("User Info")
    st.info("System Ready")

# Tabs
tab1, tab2, tab3 = st.tabs(["⚙️ Configuration", "📄 Document Upload", "💬 Chat Interface"])

# --- Tab 1: Configuration ---
with tab1:
    st.header("Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        api_key = st.text_input("Google AI Studio API Key", type="password", help="Get your key from makersuite.google.com")
        if api_key:
            st.session_state['api_key'] = api_key
            
    with col2:
        chunk_size = st.slider("Chunk Size", 200, 2000, DEFAULT_CHUNK_SIZE)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, DEFAULT_CHUNK_OVERLAP)
        
    k_retrieval_slider = st.slider("Number of relevant chunks to retrieve", 1, 10, 5)
    
    if st.button("Reset Vector Database", type="primary"):
        clear_collection()
        st.success("Database cleared!")
        time.sleep(1)
        st.rerun()
        
    coll = get_collection()
    st.markdown("### Current Status")
    st.metric("Total Documents Indexed", f"{coll.count()}")
    
    if 'api_key' in st.session_state:
        st.success("API Key is set")
    else:
        st.warning("Please provide an API Key")

# --- Tab 2: Document Upload ---
with tab2:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing PDFs..."):
                results = process_and_index_documents(uploaded_files, chunk_size, chunk_overlap)
                
            st.write("### Processing Results")
            for res in results:
                if "Success" in res['status']:
                    st.success(f"📄 {res['name']}: {res['pages']} pages, {res['chunks']} chunks generated.")
                else:
                    st.error(f"❌ {res['name']}: {res['status']}")

    st.divider()
    st.subheader("Knowledge Base")
    try:
        coll = get_collection()
        count = coll.count()
        if count > 0:
            st.info(f"Contains {count} chunks total.")
    except:
        pass

# --- Tab 3: Chat Interface ---
with tab3:
    st.header("Chat with your PDFs")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "expanded_context" in message:
                with st.expander("View Retrieved Context"):
                    for i, ctx in enumerate(message["expanded_context"]):
                        st.markdown(f"**Chunk {i+1} (Score: {ctx['hybrid_score']:.3f})**")
                        st.markdown(f"*Source: {ctx['metadata'].get('source_file')} (Page {ctx['metadata'].get('page_number')})*")
                        st.text(ctx['text'])
                        st.divider()

    if prompt := st.chat_input("Ask a question..."):
        if 'api_key' not in st.session_state:
            st.error("Please enter your Google API Key in the Configuration tab.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    relevant_chunks = hybrid_retrieval(prompt, k_retrieval_slider, 0, st.session_state['api_key'])
                    
                    if not relevant_chunks:
                        response_text = "I couldn't find any relevant information in the uploaded documents."
                        st.markdown(response_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                    else:
                        response_text = generate_rag_response(prompt, relevant_chunks, st.session_state['api_key'])
                        st.markdown(response_text)
                        
                        with st.expander("View Retrieved Context"):
                            for i, ctx in enumerate(relevant_chunks):
                                st.markdown(f"**Chunk {i+1} (Score: {ctx['hybrid_score']:.3f})**")
                                st.markdown(f"*Source: {ctx['metadata'].get('source_file')} (Page {ctx['metadata'].get('page_number')})*")
                                st.text(ctx['text'])
                                st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "expanded_context": relevant_chunks
                        })

if __name__ == "__main__":
    pass
