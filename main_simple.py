# ===== main_simple.py =====
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

# Vector store and embeddings (using scikit-learn instead of FAISS)
import numpy as np
import openai
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

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
    "top_k": int(os.getenv("TOP_K", "8")),
    "temperature": float(os.getenv("TEMPERATURE", "0.3")),
    "max_tokens": int(os.getenv("MAX_TOKENS", "2000"))
}

print(CONFIG)

class DocumentChunk:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata
        self.id = str(uuid.uuid4())

class SimpleVectorStore:
    def __init__(self, embedding_dim: int = 1536, store_path: str = "vector_store"):
        self.embedding_dim = embedding_dim
        self.store_path = store_path
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
        
        # Save after adding documents
        self.save()
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[tuple]:
        """Search for similar documents using cosine similarity"""
        if len(self.embeddings) == 0:
            return []
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.embeddings)
        query_array = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_array, embeddings_array)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.documents):
                # Convert similarity to distance (1 - similarity)
                distance = 1 - similarities[idx]
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
            docs_path = os.path.join(self.store_path, "documents.pkl")
            emb_path = os.path.join(self.store_path, "embeddings.pkl")
            
            if os.path.exists(docs_path) and os.path.exists(emb_path):
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
            self.documents = []
            self.embeddings = []

class ChatBot:
    def __init__(self):
        self.vector_store = SimpleVectorStore()
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
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_endings = ['.', '!', '?', '\n\n']
                for ending in sentence_endings:
                    last_ending = text.rfind(ending, start, end)
                    if last_ending > start + chunk_size // 2:  # Only break if it's not too early
                        end = last_ending + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position, accounting for overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a list of texts using OpenAI"""
        try:
            response = client.embeddings.create(
                model=CONFIG["embedding_model"],
                input=texts
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Return random embeddings as fallback
            return np.random.rand(len(texts), self.vector_store.embedding_dim)
        
    def enhance_query_for_rag(self, user_query: str) -> List[str]:
        """Enhance user query for better RAG retrieval"""
        enhanced_queries = [user_query]
        
        # Add variations for better retrieval
        if "investment" in user_query.lower():
            enhanced_queries.extend([
                f"investment strategy {user_query}",
                f"portfolio allocation {user_query}",
                f"asset allocation {user_query}"
            ])
        
        if "risk" in user_query.lower():
            enhanced_queries.extend([
                f"risk tolerance {user_query}",
                f"risk management {user_query}",
                f"risk assessment {user_query}"
            ])
            
        return enhanced_queries[:3]  # Limit to 3 queries
        
    async def process_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Process uploaded document and add to vector store"""
        try:
            # Extract text from file
            text = self.extract_text_from_file(file_content, filename)
            
            if not text.strip():
                return {
                    "success": False,
                    "message": "No text content found in the file"
                }
            
            # Chunk the text
            chunks = self.chunk_text(text, CONFIG["chunk_size"], CONFIG["chunk_overlap"])
            
            # Create document chunks with metadata
            document_chunks = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "filename": filename,
                    "chunk_id": i,
                    "upload_time": datetime.now().isoformat(),
                    "content_preview": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                document_chunks.append(DocumentChunk(chunk, metadata))
            
            # Get embeddings for chunks
            chunk_texts = [doc.content for doc in document_chunks]
            embeddings = self.get_embeddings(chunk_texts)
            
            # Add to vector store
            self.vector_store.add_documents(document_chunks, embeddings)
            
            return {
                "success": True,
                "message": f"Successfully processed {filename}",
                "chunks_created": len(chunks),
                "filename": filename
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing document: {str(e)}"
            }
        
    async def generate_response(self, user_query: str) -> Dict[str, Any]:
        """Generate AI response using RAG"""
        try:
            # Enhance query for better retrieval
            enhanced_queries = self.enhance_query_for_rag(user_query)
            
            # Search for relevant documents
            search_results = []
            for query in enhanced_queries:
                query_embedding = self.get_embeddings([query])[0]
                results = self.vector_store.search(query_embedding, CONFIG["top_k"])
                search_results.extend(results)
            
            # Remove duplicates and get top results
            seen_docs = set()
            unique_results = []
            for doc, distance in search_results:
                if doc.id not in seen_docs:
                    seen_docs.add(doc.id)
                    unique_results.append((doc, distance))
                    if len(unique_results) >= CONFIG["top_k"]:
                        break
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for doc, distance in unique_results:
                context_parts.append(f"Document: {doc.metadata['filename']}\nContent: {doc.content}")
                sources.append({
                    "filename": doc.metadata["filename"],
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "relevance_score": 1 - distance
                })
            
            context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
            
            # Generate AI response
            system_prompt = """You are RudoWealth, an AI-powered investment advisor. You help users with:

1. Investment strategy recommendations based on their profile
2. Portfolio allocation advice
3. Risk assessment and management
4. Market analysis and insights

When providing recommendations:
- Be specific and actionable
- Reference the provided documents when relevant
- Consider user risk tolerance and investment goals
- Provide clear explanations for your recommendations
- Use professional but accessible language

If no relevant documents are provided, provide general investment advice based on best practices."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nUser Question: {user_query}"}
            ]
            
            response = client.chat.completions.create(
                model=CONFIG["llm_model"],
                messages=messages,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"]
            )
            
            ai_response = response.choices[0].message.content
            
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
        content = await file.read()
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
        "vector_store_size": len(chatbot.vector_store.embeddings)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 