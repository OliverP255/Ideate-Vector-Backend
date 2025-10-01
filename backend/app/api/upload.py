"""
Document upload API endpoints.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import PyPDF2
import fitz  # PyMuPDF
# Mock SentenceTransformer for testing
class SentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts):
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 384).tolist()
import numpy as np

from ..services.search import SearchService

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory storage for session documents (not persistent)
session_documents: Dict[str, List[Dict[str, Any]]] = {}

@router.post("/upload/document")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = "default_user"
):
    """
    Upload a PDF document and generate embedding for session-only use.
    
    Args:
        file: PDF file to upload
        user_id: User ID for session management
        
    Returns:
        Document data with embedding and coordinates
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        content = await file.read()
        
        # Extract text from PDF
        text = extract_text_from_pdf_content(content)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Generate document ID
        document_id = f"upload_{user_id}_{len(session_documents.get(user_id, []))}"
        
        # Generate embedding
        embedding_model = SentenceTransformer('all-MiniLM-L12-v2')
        title = file.filename.replace('.pdf', '')
        combined_text = f"{title}\n\n{text}"
        embedding = embedding_model.encode(combined_text)
        
        # Generate random coordinates for the map (in a reasonable range)
        import random
        random.seed(hash(document_id))  # Deterministic based on document ID
        x = random.uniform(-15, 15)
        y = random.uniform(-15, 15)
        
        # Create document data
        document = {
            'document_id': document_id,
            'title': title,
            'content': text[:1000] + "..." if len(text) > 1000 else text,  # Truncate for response
            'full_content': text,
            'embedding': embedding.tolist(),
            'coordinates': [x, y],
            'source': 'upload',
            'user_id': user_id,
            'upload_time': str(__import__('datetime').datetime.now())
        }
        
        # Store in session
        if user_id not in session_documents:
            session_documents[user_id] = []
        session_documents[user_id].append(document)
        
        logger.info(f"Uploaded document {document_id} for user {user_id}")
        
        return {
            "success": True,
            "document": document,
            "message": f"Document '{title}' uploaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/upload/documents/{user_id}")
async def get_user_documents(user_id: str):
    """
    Get all uploaded documents for a user session.
    
    Args:
        user_id: User ID
        
    Returns:
        List of user's uploaded documents
    """
    try:
        documents = session_documents.get(user_id, [])
        
        # Return only essential data (not full embeddings)
        response_docs = []
        for doc in documents:
            response_docs.append({
                'document_id': doc['document_id'],
                'title': doc['title'],
                'content': doc['content'],
                'coordinates': doc['coordinates'],
                'source': doc['source'],
                'upload_time': doc['upload_time']
            })
        
        return {
            "documents": response_docs,
            "count": len(response_docs)
        }
        
    except Exception as e:
        logger.error(f"Error getting user documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@router.delete("/upload/documents/{user_id}")
async def clear_user_documents(user_id: str):
    """
    Clear all uploaded documents for a user session.
    
    Args:
        user_id: User ID
        
    Returns:
        Success message
    """
    try:
        if user_id in session_documents:
            del session_documents[user_id]
        
        return {"success": True, "message": f"Cleared documents for user {user_id}"}
        
    except Exception as e:
        logger.error(f"Error clearing user documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

def extract_text_from_pdf_content(content: bytes) -> str:
    """Extract text from PDF content."""
    try:
        # Try PyMuPDF first
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        
        if text.strip():
            return clean_text(text)
        
        # Fallback to PyPDF2
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        return clean_text(text) if text.strip() else None
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common PDF artifacts
    text = text.replace('\x00', '')  # Null bytes
    text = text.replace('\ufeff', '')  # BOM
    
    # Limit text length (keep first 50k characters for embedding)
    if len(text) > 50000:
        text = text[:50000] + "..."
    
    return text
