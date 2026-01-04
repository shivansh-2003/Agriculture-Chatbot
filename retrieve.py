#!/usr/bin/env python3
"""
Phase 3: RAG Implementation
Retrieval and Response Generation for Agriculture Chatbot

This module implements:
1. Semantic search in vector stores using similarity metrics
2. Retrieve top-k relevant chunks (k=3-5)
3. Re-rank results for relevance (optional)
4. Context assembly from retrieved chunks
5. Generate farmer-friendly responses using LLM with retrieved context
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_community.document_compressors import FlashrankRerank

# Load environment variables
load_dotenv()

# Configuration
OPENAI_MODEL = "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone index names (from ingest.py)
PINECONE_INDEX_NAME_CITRUS = "citrus-diseases-pests"
PINECONE_INDEX_NAME_SCHEMES = "government-schemes"

# OpenAI Embedding model
EMBEDDING_MODEL = "text-embedding-ada-002"

# Retrieval parameters
K_RETRIEVAL = 5  # Number of chunks to retrieve initially
K_RERANK = 3  # Number of top chunks after reranking
USE_RERANKING = True  # Enable reranking with FlashRank


class RetrievalSystem:
    """Retrieval system for agriculture chatbot using Pinecone vector stores."""
    
    def __init__(self):
        """Initialize the retrieval system with vector stores and LLM."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        )
        
        # Initialize vector stores
        self.citrus_vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME_CITRUS,
            embedding=self.embeddings
        )
        
        self.schemes_vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME_SCHEMES,
            embedding=self.embeddings
        )
        
        # Initialize reranker if enabled
        self.compressor = None
        self.use_reranking = USE_RERANKING
        if self.use_reranking:
            try:
                from flashrank import Ranker
                # Initialize FlashRank Ranker
                ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
                # Initialize FlashrankRerank compressor
                self.compressor = FlashrankRerank(client=ranker, top_n=K_RERANK)
                print("✅ FlashRank reranker initialized successfully")
            except (ImportError, Exception) as e:
                print(f"Warning: FlashrankRerank not available. Reranking disabled. Error: {e}")
                self.use_reranking = False
    
    def retrieve_disease(self, query: str, k: int = K_RETRIEVAL) -> List[Document]:
        """Retrieve relevant documents from Citrus Pests & Diseases knowledge base.
        
        Args:
            query: Search query
            k: Number of documents to retrieve initially (before reranking)
            
        Returns:
            List of retrieved documents (reranked if enabled)
        """
        # Create base retriever
        base_retriever = self.citrus_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Retrieve documents
        docs = base_retriever.invoke(query)
        
        # Apply reranking if enabled
        if self.compressor and self.use_reranking:
            # Use compressor to rerank documents
            docs = self.compressor.compress_documents(documents=docs, query=query)
        
        return docs
    
    def retrieve_scheme(self, query: str, k: int = K_RETRIEVAL) -> List[Document]:
        """Retrieve relevant documents from Government Schemes knowledge base.
        
        Args:
            query: Search query
            k: Number of documents to retrieve initially (before reranking)
            
        Returns:
            List of retrieved documents (reranked if enabled)
        """
        # Create base retriever
        base_retriever = self.schemes_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Retrieve documents
        docs = base_retriever.invoke(query)
        
        # Apply reranking if enabled
        if self.compressor and self.use_reranking:
            # Use compressor to rerank documents
            docs = self.compressor.compress_documents(documents=docs, query=query)
        
        return docs
    
    def retrieve_hybrid(
        self, 
        disease_query: str, 
        scheme_query: str, 
        k: int = K_RETRIEVAL
    ) -> Dict[str, List[Document]]:
        """Retrieve documents from both knowledge bases for hybrid queries.
        
        Args:
            disease_query: Query for disease knowledge base
            scheme_query: Query for scheme knowledge base
            k: Number of documents to retrieve from each base
            
        Returns:
            Dictionary with 'disease_docs' and 'scheme_docs' keys
        """
        disease_docs = self.retrieve_disease(disease_query, k)
        scheme_docs = self.retrieve_scheme(scheme_query, k)
        
        return {
            "disease_docs": disease_docs,
            "scheme_docs": scheme_docs
        }
    
    def format_documents(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string.
        
        Args:
            docs: List of Document objects
            
        Returns:
            Formatted context string
        """
        if not docs:
            return "No relevant information found."
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            
            formatted_parts.append(f"[Source {i}: {source}]\n{content}\n")
        
        return "\n".join(formatted_parts)
    
    # LLM response generation temporarily disabled
    # def generate_response(...) - will be re-enabled later
    
    def retrieve_and_generate(
        self,
        query: str,
        intent: str,
        disease_query: Optional[str] = None,
        scheme_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Retrieve documents from vector database (with reranking).
        Note: LLM response generation is temporarily disabled.
        
        Args:
            query: Original user query
            intent: Query intent (disease, scheme, or hybrid)
            disease_query: Disease-focused query (for hybrid)
            scheme_query: Scheme-focused query (for hybrid)
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        disease_docs = None
        scheme_docs = None
        context = ""
        
        if intent == "disease":
            disease_docs = self.retrieve_disease(query)
            context = self.format_documents(disease_docs)
            
        elif intent == "scheme":
            scheme_docs = self.retrieve_scheme(query)
            context = self.format_documents(scheme_docs)
            
        elif intent == "hybrid":
            if not disease_query or not scheme_query:
                raise ValueError("Both disease_query and scheme_query are required for hybrid intent")
            
            retrieved = self.retrieve_hybrid(disease_query, scheme_query)
            disease_docs = retrieved["disease_docs"]
            scheme_docs = retrieved["scheme_docs"]
            
            # Format both contexts for hybrid
            disease_context = self.format_documents(disease_docs)
            scheme_context = self.format_documents(scheme_docs)
            context = f"DISEASE/PEST INFORMATION:\n{disease_context}\n\nGOVERNMENT SCHEMES INFORMATION:\n{scheme_context}"
        else:
            raise ValueError(f"Unknown intent: {intent}")
        
        return {
            "response": None,  # LLM response generation disabled for now
            "disease_docs": disease_docs,
            "scheme_docs": scheme_docs,
            "context": context,
            "num_disease_docs": len(disease_docs) if disease_docs else 0,
            "num_scheme_docs": len(scheme_docs) if scheme_docs else 0
        }


def main():
    """Test the retrieval system."""
    print("🚀 Retrieval System Test")
    print("="*60)
    
    retrieval_system = RetrievalSystem()
    
    # Test queries
    test_cases = [
        {
            "query": "What is citrus canker?",
            "intent": "disease",
            "disease_query": None,
            "scheme_query": None
        },
        {
            "query": "What government schemes are available for citrus farmers?",
            "intent": "scheme",
            "disease_query": None,
            "scheme_query": None
        },
        {
            "query": "Can I get government support for drip irrigation to prevent root diseases?",
            "intent": "hybrid",
            "disease_query": "How does drip irrigation prevent root diseases in citrus?",
            "scheme_query": "What government subsidies are available for drip irrigation?"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['query']}")
        print(f"Intent: {test_case['intent']}")
        print("-"*60)
        
        try:
            result = retrieval_system.retrieve_and_generate(
                query=test_case["query"],
                intent=test_case["intent"],
                disease_query=test_case.get("disease_query"),
                scheme_query=test_case.get("scheme_query")
            )
            
            print(f"\n📄 Response:")
            print(result["response"])
            print(f"\n📊 Metadata:")
            print(f"  Disease docs: {result['num_disease_docs']}")
            print(f"  Scheme docs: {result['num_scheme_docs']}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Retrieval System Test Complete")


if __name__ == "__main__":
    main()

