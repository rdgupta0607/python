I would like you to generate a fully working Streamlit app that implements a RAG (Retrieval Augmented Generation) chatbot for PDF documents. Please ensure that:
CORE REQUIREMENTS:

The code runs correctly the first time without syntax or logic errors.
Use Google gemini-2.5-flash as the LLM (via google.generativeai package).
Use HuggingFace sentence-transformers for embeddings - specifically use the all-MiniLM-L6-v2 model which is already downloaded locally and will be available in the folder all-MiniLM-L12-v2 which is a sub folder where the main streamlit app code is there. Do NOT use any API-based embedding services.
Use ChromaDB as the vector store - use persistent local storage so embeddings survive app restarts.


APP LAYOUT - THREE TABS:
Tab 1: ⚙️ Configuration

API Key input field (Google AI Studio API key)
Chunk size slider (range: 200-2000 characters, default: 1000)
Chunk overlap slider (range: 0-500 characters, default: 200)
Number of relevant chunks to retrieve slider (range: 1-10, default: 5)
Button to clear/reset the vector database
Display current configuration status

Tab 2: 📄 Document Upload

File uploader that accepts multiple PDF files simultaneously
Show upload progress for each file
After upload, display:

List of uploaded documents with page counts
Total chunks created
Processing status (success/failure for each document)


Store these metadata with each chunk in ChromaDB:

source_file: original PDF filename
page_number: which page the chunk came from
chunk_index: position of chunk within the document
upload_timestamp: when the document was processed



Tab 3: 💬 Chat Interface

Clean chat interface with message history
User input at the bottom
For each response, display:

The AI's answer
Source citations showing which documents and pages were used
Expandable section showing the retrieved chunks
Confidence/relevance scores for retrieved chunks




TECHNICAL IMPLEMENTATION:
PDF Processing:

Use PyPDF2 or pypdf to extract text from PDFs
Handle multi-page documents properly
Implement proper error handling for corrupted or password-protected PDFs

Chunking Strategy:

Implement recursive character text splitting
Respect sentence boundaries where possible
Use the configurable chunk size and overlap from Tab 1
Ensure no chunk is empty or contains only whitespace

Embedding Generation:

Use sentence-transformers library
Model: all-MiniLM-L6-v2 (384 dimensions)
Generate embeddings locally (no API calls for embeddings)
Show embedding generation progress

ChromaDB Vector Store:

Create a persistent ChromaDB collection
Store embeddings with full metadata
Implement similarity search using cosine similarity
Collection name should be configurable or use a sensible default like "pdf_documents"

RAG Query Flow:

User enters a question
Generate embedding for the question using the same model
Perform similarity search in ChromaDB to retrieve top-k relevant chunks
Construct a prompt that includes:

System instructions for the LLM

Retrieved context chunks with their source information
The user's question
Instructions to cite sources in the response

Retrieval MUST use hybrid + reranking for broad questions:

Perform semantic retrieval with embeddings from ChromaDB (cosine).

Retrieve a large candidate set (fetch_k 80–150).

Apply a hybrid rerank that combines semantic similarity with a lexical overlap score (keyword boosting) to avoid missing exact technical terms.

Add an optional “LLM rerank” step (Gemini) to select the best 8–12 chunks for final answering.

Do NOT pass more than 12 chunks into the final answer prompt; use reranking to select them.

Implement dynamic thresholding (if not enough chunks pass, lower threshold to a safe minimum like 0.55).

Chunking should support page ranges (page_start, page_end) so that context spanning pages is preserved.



Send to Gemini gemini-2.5-flash for response generation
Display response with proper source attribution

Prompt Template for LLM:
You are a helpful assistant that answers questions based on the provided document context.

CONTEXT FROM DOCUMENTS:
{retrieved_chunks_with_sources}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so
- Cite your sources by mentioning the document name and page number
- Be concise but comprehensive

UI/UX REQUIREMENTS:

Professional, modern design with consistent styling
Use Streamlit's native theming capabilities
Add appropriate icons and emojis for visual appeal
Show loading spinners during processing
Display success/error toast messages
Responsive layout that works on different screen sizes
Use st.columns for side-by-side layouts where appropriate
Add helpful tooltips and descriptions for configuration options
Implement session state properly to maintain chat history


ERROR HANDLING:

Validate API key before allowing queries
Handle PDF extraction failures gracefully
Show user-friendly error messages
Implement retry logic for LLM API calls
Handle empty or invalid queries
Validate that documents are uploaded before allowing chat


ADDITIONAL FEATURES:

Display total documents in the knowledge base
Show storage/memory usage statistics
Allow users to see all documents in the knowledge base
Option to remove specific documents from the vector store
Export chat history functionality


GENERATE requirements.txt
Create a requirements.txt file that includes ALL necessary packages with compatible versions:
streamlit>=1.28.0
google-generativeai>=0.3.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
pypdf>=3.0.0
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0

CODE QUALITY:

Add clear comments explaining each major section
Use meaningful variable and function names
Organize code into logical functions
Follow Python best practices
Ensure all imports are at the top of the file
No hardcoded values - use configuration variables


IMPORTANT NOTES:

The embedding model (all-MiniLM-L6-v2) will be downloaded automatically on first run - this is expected behavior
ChromaDB data should persist in a local folder (e.g., "./chroma_db")
The app should work offline for embeddings (only LLM calls need internet)
Test with various PDF types and sizes

The code you generate needs to be 100% perfect on the first attempt. No errors in any functionality. Generate the complete app.py file and requirements.txt.