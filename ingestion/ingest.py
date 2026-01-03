#!/usr/bin/env python3
"""
Phase 1: Document Processing & Vector Store Creation
Ingestion pipeline for Agriculture Chatbot

This script:
1. Loads and parses PDF documents
2. Implements intelligent text chunking (500-1000 tokens with overlap)
3. Generates embeddings using OpenAI or other embedding models
4. Stores embeddings in Pinecone vector database
5. Creates separate collections for:
   - Citrus Pests & Diseases knowledge base
   - Government Schemes knowledge base
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import tqdm

# Load environment variables
load_dotenv()

# Configuration
CHUNK_SIZE = 1000  # tokens (approximately 1000 chars for English)
CHUNK_OVERLAP = 200  # tokens for overlap
EMBEDDING_MODEL = "text-embedding-3-small"  # or "text-embedding-ada-002"

# PDF documents
PDF_DIR = Path(__file__).parent.parent
CITRUS_PDF = PDF_DIR / "CitrusPlantPestsAndDiseases.pdf"
SCHEMES_PDF = PDF_DIR / "GovernmentSchemes.pdf"

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME_CITRUS = "citrus-diseases-pests"
PINECONE_INDEX_NAME_SCHEMES = "government-schemes"

def initialize_pinecone() -> Pinecone:
    """Initialize Pinecone client."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc

def create_pinecone_index(pc: Pinecone, index_name: str, dimension: int = 1536):
    """Create Pinecone index if it doesn't exist."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Index {index_name} created successfully")
        # Wait a moment for index to be ready
        import time
        time.sleep(2)
    else:
        print(f"Index {index_name} already exists")

def load_pdf(pdf_path: Path) -> List[Any]:
    """Load PDF document using LangChain PyPDFLoader."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    print(f"Loading PDF: {pdf_path.name}")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from {pdf_path.name}")
    
    return documents

def chunk_documents(documents: List[Any], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Any]:
    """Chunk documents using RecursiveCharacterTextSplitter."""
    print(f"Chunking documents with size={chunk_size}, overlap={chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    return chunks

def add_metadata_to_chunks(chunks: List[Any], source_name: str, collection_type: str) -> List[Any]:
    """Add metadata to document chunks."""
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "source": source_name,
            "collection": collection_type,
            "chunk_id": i,
            "total_chunks": len(chunks)
        })
        # Add page number if available
        if hasattr(chunk, 'metadata') and 'page' in chunk.metadata:
            chunk.metadata['page_number'] = chunk.metadata['page']
    
    return chunks

def ingest_documents(
    pdf_path: Path,
    index_name: str,
    collection_type: str,
    embeddings: OpenAIEmbeddings,
    pc: Pinecone
):
    """Ingest documents into Pinecone."""
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path.name}")
    print(f"Collection: {collection_type}")
    print(f"Index: {index_name}")
    print(f"{'='*60}\n")
    
    # Load PDF
    documents = load_pdf(pdf_path)
    
    # Chunk documents
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    
    # Add metadata
    chunks = add_metadata_to_chunks(chunks, pdf_path.name, collection_type)
    
    # Create or get index
    create_pinecone_index(pc, index_name, dimension=1536)  # OpenAI embeddings dimension
    
    # Initialize Pinecone vector store
    print(f"Initializing Pinecone vector store for {index_name}...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    
    print(f"✅ Successfully ingested {len(chunks)} chunks into {index_name}")
    return vectorstore

def main():
    """Main ingestion function."""
    print("🚀 Starting Document Ingestion Pipeline")
    print("="*60)
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    
    # Initialize embeddings
    print("Initializing embeddings model...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    
    # Initialize Pinecone
    print("Initializing Pinecone client...")
    pc = initialize_pinecone()
    
    # Ingest Citrus Pests & Diseases PDF
    citrus_vectorstore = ingest_documents(
        pdf_path=CITRUS_PDF,
        index_name=PINECONE_INDEX_NAME_CITRUS,
        collection_type="citrus_diseases_pests",
        embeddings=embeddings,
        pc=pc
    )
    
    # Ingest Government Schemes PDF
    schemes_vectorstore = ingest_documents(
        pdf_path=SCHEMES_PDF,
        index_name=PINECONE_INDEX_NAME_SCHEMES,
        collection_type="government_schemes",
        embeddings=embeddings,
        pc=pc
    )
    
    print("\n" + "="*60)
    print("✅ Ingestion Pipeline Completed Successfully!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  - Citrus Diseases & Pests: Index '{PINECONE_INDEX_NAME_CITRUS}'")
    print(f"  - Government Schemes: Index '{PINECONE_INDEX_NAME_SCHEMES}'")
    print(f"\nYou can now use these indexes for RAG queries.")

if __name__ == "__main__":
    main()

