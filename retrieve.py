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
from langsmith import traceable
from response import ResponseGenerator

# Load environment variables
load_dotenv()

# Configuration
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model for LLM
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
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize OpenAI embeddings (keeping OpenAI for embeddings)
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
        
        # Initialize response generator (using OpenAI for LLM)
        self.response_generator = ResponseGenerator(
            model_name=OPENAI_MODEL,
            temperature=0.0
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
    
    @traceable(name="retrieval_system.retrieve_disease")
    def retrieve_disease(self, query: str, k: int = K_RETRIEVAL) -> List[Document]:
        """Retrieve relevant documents from Citrus Pests & Diseases knowledge base.
        
        Args:
            query: Search query
            k: Number of documents to retrieve initially (before reranking)
            
        Returns:
            List of retrieved documents (reranked if enabled) with similarity scores in metadata
        """
        # Use similarity_search_with_score to get scores
        results_with_scores = self.citrus_vectorstore.similarity_search_with_score(
            query,
            k=k
        )
        
        # Convert to Document objects with scores in metadata
        docs = []
        for doc, score in results_with_scores:
            # Pinecone returns cosine similarity scores (0-1 range for normalized embeddings)
            # Use score directly as confidence (already normalized to 0-1)
            confidence = float(score)
            
            # Add score/confidence to metadata
            doc.metadata["similarity_score"] = float(score)
            doc.metadata["confidence"] = confidence
            docs.append(doc)
        
        # Apply reranking if enabled
        if self.compressor and self.use_reranking:
            # Use compressor to rerank documents
            docs = self.compressor.compress_documents(documents=docs, query=query)
            # Note: Reranking may change order, but original scores are still in metadata
        
        return docs
    
    @traceable(name="retrieval_system.retrieve_scheme")
    def retrieve_scheme(self, query: str, k: int = K_RETRIEVAL) -> List[Document]:
        """Retrieve relevant documents from Government Schemes knowledge base.
        
        Args:
            query: Search query
            k: Number of documents to retrieve initially (before reranking)
            
        Returns:
            List of retrieved documents (reranked if enabled) with similarity scores in metadata
        """
        # Use similarity_search_with_score to get scores
        results_with_scores = self.schemes_vectorstore.similarity_search_with_score(
            query,
            k=k
        )
        
        # Convert to Document objects with scores in metadata
        docs = []
        for doc, score in results_with_scores:
            # Pinecone returns cosine similarity scores (0-1 range for normalized embeddings)
            # Use score directly as confidence (already normalized to 0-1)
            confidence = float(score)
            
            # Add score/confidence to metadata
            doc.metadata["similarity_score"] = float(score)
            doc.metadata["confidence"] = confidence
            docs.append(doc)
        
        # Apply reranking if enabled
        if self.compressor and self.use_reranking:
            # Use compressor to rerank documents
            docs = self.compressor.compress_documents(documents=docs, query=query)
            # Note: Reranking may change order, but original scores are still in metadata
        
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
        """Format retrieved documents into context string with citations.
        
        Args:
            docs: List of Document objects with metadata (including confidence and page_number)
            
        Returns:
            Formatted context string with citations including page numbers and confidence scores
        """
        if not docs:
            return "No relevant information found."
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            metadata = doc.metadata
            source = metadata.get("source", "Unknown")
            page_number = metadata.get("page_number") or metadata.get("page")
            confidence = metadata.get("confidence", 0.0)
            
            # Format citation with page number and confidence
            citation_parts = [f"Source {i}: {source}"]
            if page_number is not None:
                citation_parts.append(f"Page {page_number}")
            citation_parts.append(f"Confidence: {confidence:.2%}")
            citation = " | ".join(citation_parts)
            
            formatted_parts.append(f"[{citation}]\n{content}\n")
        
        return "\n".join(formatted_parts)
    
    @traceable(name="retrieval_system.retrieve_and_generate")
    def retrieve_and_generate(
        self,
        query: str,
        intent: str,
        disease_query: Optional[str] = None,
        scheme_query: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve documents and generate response.
        
        Args:
            query: Original user query
            intent: Query intent (disease, scheme, or hybrid)
            disease_query: Disease-focused query (for hybrid)
            scheme_query: Scheme-focused query (for hybrid)
            conversation_history: Optional conversation history for context
            
        Returns:
            Dictionary with response, retrieved documents, and metadata
        """
        disease_docs = None
        scheme_docs = None
        context = ""
        
        if intent == "disease":
            disease_docs = self.retrieve_disease(query)
            context = self.format_documents(disease_docs)
            response = self.response_generator.generate_response(
                query=query,
                context=context,
                intent=intent,
                disease_docs=disease_docs,
                conversation_history=conversation_history
            )
            
        elif intent == "scheme":
            scheme_docs = self.retrieve_scheme(query)
            context = self.format_documents(scheme_docs)
            response = self.response_generator.generate_response(
                query=query,
                context=context,
                intent=intent,
                scheme_docs=scheme_docs,
                conversation_history=conversation_history
            )
            
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
            
            response = self.response_generator.generate_response(
                query=query,
                context=context,
                intent=intent,
                disease_docs=disease_docs,
                scheme_docs=scheme_docs,
                conversation_history=conversation_history
            )
        else:
            raise ValueError(f"Unknown intent: {intent}")
        
        return {
            "response": response,
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

