# ===== main.py =====
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime
import logging
from pathlib import Path

# Document processing
import PyPDF2
import docx
from io import BytesIO
import pandas as pd

# Vector store and embeddings
import faiss
import numpy as np
import openai
from openai import OpenAI

# Utilities
import hashlib
import pickle
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RudoWealth AI Chatbot", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
CONFIG = {
    "embedding_model": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    "llm_model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
    "top_k": int(os.getenv("TOP_K", "8")),  # Increased for better retrieval
    "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "2000"))
}

print(CONFIG)

class DocumentChunk:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.id = str(uuid.uuid4())

class VectorStore:
    def __init__(self, embedding_dim: int = 1536, store_path: str = "vector_store"):
        self.embedding_dim = embedding_dim
        self.store_path = store_path
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.embeddings = []
        
        # Create directory if it doesn't exist
        os.makedirs(store_path, exist_ok=True)
        
        # Load existing data if available
        self.load()
        
    def add_documents(self, documents: List[DocumentChunk], embeddings: np.ndarray):
        """Add documents and their embeddings to the vector store"""
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        # Save after adding documents
        self.save()
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            return []
        
        distances, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), k)
        results = []
        
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], distance))
        
        return results
    
    def search_by_filename(self, filename_pattern: str, limit: int = 3) -> List[DocumentChunk]:
        """Search for documents by filename pattern"""
        matching_docs = []
        for doc in self.documents:
            if filename_pattern.lower() in doc.metadata["filename"].lower():
                matching_docs.append(doc)
                if len(matching_docs) >= limit:
                    break
        return matching_docs
    
    def save(self):
        """Save vector store to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, os.path.join(self.store_path, "faiss_index.bin"))
            
            # Save documents and embeddings
            with open(os.path.join(self.store_path, "documents.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            
            with open(os.path.join(self.store_path, "embeddings.pkl"), "wb") as f:
                pickle.dump(self.embeddings, f)
                
            logger.info(f"Vector store saved with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
    
    def load(self):
        """Load vector store from disk"""
        try:
            index_path = os.path.join(self.store_path, "faiss_index.bin")
            docs_path = os.path.join(self.store_path, "documents.pkl")
            emb_path = os.path.join(self.store_path, "embeddings.pkl")
            
            if os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(emb_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load documents and embeddings
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                
                with open(emb_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                    
                logger.info(f"Vector store loaded with {len(self.documents)} documents")
            else:
                logger.info("No existing vector store found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            # Reset to empty state if loading fails
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.documents = []
            self.embeddings = []

class ChatBot:
    def __init__(self):
        self.vector_store = VectorStore()
        self.chat_history = []
        
    def extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text from various file formats"""
        try:
            if filename.endswith('.pdf'):
                return self._extract_from_pdf(file_content)
            elif filename.endswith('.docx'):
                return self._extract_from_docx(file_content)
            elif filename.endswith('.txt'):
                return file_content.decode('utf-8')
            elif filename.endswith('.csv'):
                return self._extract_from_csv(file_content)
            elif filename.endswith('.json'):
                return self._extract_from_json(file_content)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise
            
    def _extract_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF"""
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
        
    def _extract_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
        
    def _extract_from_csv(self, content: bytes) -> str:
        """Extract text from CSV"""
        df = pd.read_csv(BytesIO(content))
        return df.to_string()
        
    def _extract_from_json(self, content: bytes) -> str:
        """Extract text from JSON"""
        data = json.loads(content.decode('utf-8'))
        return json.dumps(data, indent=2)
        
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Find the last sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:
                    chunk = text[start:boundary + 1]
                    end = boundary + 1
                    
            chunks.append(chunk.strip())
            start = end - overlap
            
        return chunks
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts using OpenAI API"""
        try:
            response = client.embeddings.create(
                model=CONFIG["embedding_model"],
                input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
            
    def enhance_query_for_rag(self, user_query: str) -> List[str]:
        """Create multiple enhanced queries for better RAG retrieval"""
        enhanced_queries = [user_query]  # Original query
        
        # Extract RUDO ID
        rudo_match = re.search(r'rudo[_-]?(\d+)', user_query, re.IGNORECASE)
        if rudo_match:
            rudo_id = f"RUDO{rudo_match.group(1).zfill(3)}"
            enhanced_queries.append(f"{rudo_id} user profile risk tolerance")
            enhanced_queries.append(f"{rudo_id} allocation strategy")
        
        # Add regime-specific queries for allocation questions
        if any(word in user_query.lower() for word in ['allocation', 'strategy', 'portfolio', 'invest']):
            enhanced_queries.append("current market regime growth investment strategy")
            enhanced_queries.append("Core Satellite Growth risk adjustments")
            enhanced_queries.append("high risk equity allocation percentage")
        
        return enhanced_queries
            
    async def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded document and add to vector store"""
        try:
            # Extract text
            text = self.extract_text_from_file(file_content, filename)
            
            # Chunk text
            chunks = self.chunk_text(text, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
            
            # Create document chunks with metadata
            doc_chunks = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "filename": filename,
                    "chunk_id": i,
                    "upload_time": datetime.now().isoformat(),
                    "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                doc_chunks.append(DocumentChunk(chunk, metadata))
            
            # Generate embeddings
            embeddings = self.get_embeddings([chunk.content for chunk in doc_chunks])
            
            # Add to vector store
            self.vector_store.add_documents(doc_chunks, embeddings)
            
            return {
                "success": True,
                "message": f"Successfully processed {filename}",
                "chunks_created": len(doc_chunks),
                "total_documents": len(self.vector_store.documents)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing {filename}: {str(e)}"
            }
            
    async def generate_response(self, user_query: str) -> Dict[str, Any]:
        """Generate response using enhanced RAG approach"""
        try:
            # Extract RUDO ID from query if present
            rudo_id = None
            rudo_match = re.search(r'rudo[_-]?(\d+)', user_query, re.IGNORECASE)
            if rudo_match:
                rudo_id = f"RUDO{rudo_match.group(1).zfill(3)}"
            
            # Create enhanced queries for better retrieval
            enhanced_queries = self.enhance_query_for_rag(user_query)
            
            # Get embeddings for all enhanced queries
            all_search_results = []
            for query in enhanced_queries:
                query_embedding = self.get_embeddings([query])[0]
                search_results = self.vector_store.search(query_embedding, CONFIG["top_k"] // len(enhanced_queries) + 2)
                all_search_results.extend(search_results)
            
            # Remove duplicates and sort by relevance
            seen_docs = set()
            unique_results = []
            for doc, distance in all_search_results:
                if doc.id not in seen_docs:
                    seen_docs.add(doc.id)
                    unique_results.append((doc, distance))
            
            # Sort by distance (lower is better) and take top results
            unique_results.sort(key=lambda x: x[1])
            search_results = unique_results[:CONFIG["top_k"]]
            
            # Force retrieve specific documents based on query type
            context = ""
            sources = []
            
            # Add RUDO ID context if found
            if rudo_id:
                context += f"TARGET USER: {rudo_id}\n\n"
                
                # Force retrieve user profile data
                user_docs = self.vector_store.search_by_filename("user_profiles", 2)
                for doc in user_docs:
                    if rudo_id in doc.content:
                        context += f"SPECIFIC USER PROFILE FOR {rudo_id}:\n{doc.content}\n\n"
                        sources.append({
                            "filename": doc.metadata["filename"],
                            "content_type": "User Profile"
                        })
                        break
            
            # Force retrieve market regime data for investment queries
            investment_keywords = ['allocation', 'strategy', 'portfolio', 'invest', 'market', 'regime']
            if any(keyword in user_query.lower() for keyword in investment_keywords):
                regime_docs = self.vector_store.search_by_filename("market_regime", 2)
                for doc in regime_docs:
                    context += f"CURRENT MARKET REGIME DATA:\n{doc.content}\n\n"
                    sources.append({
                        "filename": doc.metadata["filename"],
                        "content_type": "Market Regime"
                    })
                    break
                
                # Force retrieve investment strategy data
                strategy_docs = self.vector_store.search_by_filename("investment_strategies", 2)
                for doc in strategy_docs:
                    context += f"INVESTMENT STRATEGY RULES:\n{doc.content}\n\n"
                    sources.append({
                        "filename": doc.metadata["filename"],
                        "content_type": "Investment Strategy"
                    })
                    break
            
            # Add other relevant search results
            context += "ADDITIONAL RELEVANT CONTEXT:\n"
            for i, (doc, distance) in enumerate(search_results[:3]):
                filename = doc.metadata["filename"]
                context += f"Source {i+1} ({filename}):\n{doc.content}\n\n"
                sources.append({
                    "filename": filename,
                    "content_type": "Search Result",
                    "relevance_score": float(distance)
                })
            
            # Log what we found for debugging
            logger.info(f"Query: {user_query}")
            logger.info(f"RUDO ID detected: {rudo_id}")
            logger.info(f"Context length: {len(context)}")
            logger.info(f"Sources found: {[s['filename'] for s in sources]}")
            
            # Generate response using OpenAI with enhanced prompt
            system_prompt = """You are RudoWealth's AI Wealth Advisor specializing in Indian markets and personalized investment strategies.

CRITICAL ACCURACY RULES:
- ONLY use exact data from the provided context
- NEVER make up user information or numbers
- If specific user data is provided, use EXACT numbers and details
- If user data is missing, clearly state "User profile not found"

ALLOCATION RESPONSE RULES FOR RUDO USERS:
When asked about allocation for a specific user (like RUDO001):
1. Find their EXACT risk tolerance from user profile (High/Moderate/Conservative)
2. Identify current market regime from market regime data (Growth/Recession/Stagflation/Recovery)
3. Use investment strategy data to find EXACT allocation percentages
4. NEVER give generic ranges - use specific percentages from documents

EXAMPLE EXPECTED FORMAT:
For RUDO001 (High Risk) in Growth Regime:
- **Equity**: 80-90% (from risk_adjustments.high_risk.equity)
- **Debt**: 10-20% (from risk_adjustments.high_risk.debt)
- **Strategy**: Momentum + Growth factors
- **Core**: 60-80%, **Satellite**: 20-40%

RESPONSE STRUCTURE:
🎯 **Executive Summary**
📊 **User Profile** (if specific user)
📈 **Market Regime Analysis**
💰 **Specific Allocation** (exact percentages)
🛡️ **Implementation Strategy**
📚 **Sources**

FORMATTING:
- Use proper markdown with **bold** and *italic*
- Show specific percentages: `85% equity`
- Use bullet points for clarity
- Include exact numbers from documents"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""
CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {user_query}

CRITICAL INSTRUCTIONS:
1. Use ONLY the exact data provided in the context above
2. If RUDO user is mentioned, find their specific profile data and use exact risk tolerance
3. Use current market regime data to determine appropriate strategy
4. Apply exact allocation percentages from investment strategy rules
5. Format response with specific numbers, not generic ranges

Provide a comprehensive answer using the exact data from the context."""}
            ]
            
            response = client.chat.completions.create(
                model=CONFIG["llm_model"],
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"]
            )
            
            ai_response = response.choices[0].message.content
            
            # Clean up response
            if ai_response:
                ai_response = str(ai_response).strip()
                ai_response = ai_response.replace('\x00', '').replace('\r', '\n')
            
            # Store in chat history
            self.chat_history.append({
                "user_query": user_query,
                "ai_response": ai_response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "success": True,
                "response": ai_response,
                "sources": sources,
                "context_used": len(search_results) > 0
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "success": False,
                "response": "I'm sorry, I encountered an error while processing your question. Please try again.",
                "sources": [],
                "context_used": False
            }

# Initialize chatbot
chatbot = ChatBot()

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    context_used: bool

# Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle single file upload and processing"""
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        result = await chatbot.process_document(content, file.filename)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(
            content={"success": False, "message": str(e)},
            status_code=500
        )

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Handle chat requests"""
    try:
        result = await chatbot.generate_response(request.message)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "response": "I'm sorry, I encountered an error. Please try again.",
                "sources": [],
                "context_used": False
            },
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "documents_indexed": len(chatbot.vector_store.documents),
        "chat_history_length": len(chatbot.chat_history)
    }

@app.get("/stats")
async def get_stats():
    """Get chatbot statistics"""
    return {
        "total_documents": len(chatbot.vector_store.documents),
        "total_chats": len(chatbot.chat_history),
        "vector_store_size": chatbot.vector_store.index.ntotal
    }

@app.get("/documents")
async def get_documents():
    """Get list of stored documents with details"""
    documents_info = []
    seen_files = set()
    
    for doc in chatbot.vector_store.documents:
        filename = doc.metadata["filename"]
        if filename not in seen_files:
            seen_files.add(filename)
            
            # Count chunks for this file
            chunks_count = sum(1 for d in chatbot.vector_store.documents if d.metadata["filename"] == filename)
            
            documents_info.append({
                "filename": filename,
                "chunks_count": chunks_count,
                "upload_time": doc.metadata["upload_time"],
                "content_preview": doc.metadata["content_preview"]
            })
    
    return {
        "documents": documents_info,
        "total_files": len(seen_files),
        "total_chunks": len(chatbot.vector_store.documents)
    }

# Debug endpoint to check document content
@app.get("/debug/search/{query}")
async def debug_search(query: str):
    """Debug endpoint to see what documents are retrieved for a query"""
    try:
        # Get query embedding
        query_embedding = chatbot.get_embeddings([query])[0]
        
        # Search for documents
        search_results = chatbot.vector_store.search(query_embedding, 5)
        
        debug_info = []
        for i, (doc, distance) in enumerate(search_results):
            debug_info.append({
                "rank": i + 1,
                "filename": doc.metadata["filename"],
                "chunk_id": doc.metadata["chunk_id"],
                "distance": float(distance),
                "content_preview": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "full_content": doc.content
            })
        
        return {
            "query": query,
            "total_results": len(debug_info),
            "results": debug_info
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Create directories
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)
    
    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=8000)