#!/usr/bin/env python3
"""
Phase 3: Response Generation
Response Generator for Agriculture Chatbot

This module generates farmer-friendly responses using LLM with retrieved context.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

# Load environment variables
load_dotenv()

# Configuration
OPENAI_MODEL = "gpt-4o-mini"  # OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ResponseGenerator:
    """Response generator for agriculture chatbot using OpenAI LLM."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.0):
        """Initialize the response generator.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4o-mini)
            temperature: Temperature for generation (default: 0.0 for consistent responses)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
    
    @traceable(name="response_generator.generate_response")
    def generate_response(
        self,
        query: str,
        context: str,
        intent: str,
        disease_docs: Optional[List[Document]] = None,
        scheme_docs: Optional[List[Document]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Generate farmer-friendly response using LLM with retrieved context.
        
        Args:
            query: Original user query
            context: Formatted context from retrieved documents
            intent: Query intent (disease, scheme, or hybrid)
            disease_docs: Disease documents (for hybrid)
            scheme_docs: Scheme documents (for hybrid)
            conversation_history: Optional conversation history for context
            
        Returns:
            Generated response string
        """
        # Create prompt template based on intent
        if intent == "hybrid":
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful agricultural assistant for farmers. Your task is to provide clear, actionable, and farmer-friendly responses based on the retrieved information from knowledge bases.

For hybrid queries (combining disease/pest information with government schemes), structure your response as follows:
1. First, provide information about the disease/pest issue
2. Then, provide information about relevant government schemes
3. Connect both aspects when applicable

Guidelines:
- Use simple, clear language suitable for farmers
- Provide actionable steps and recommendations
- Cite information from sources when relevant (e.g., "According to [Source 1]...")
- Be specific and accurate
- If information is not available, say so clearly
- Focus on practical, field-ready advice
- Use numbered lists or bullet points for clarity when appropriate

Respond in a helpful, conversational tone."""),
                ("human", """User Query: {query}

DISEASE/PEST INFORMATION:
{disease_context}

GOVERNMENT SCHEMES INFORMATION:
{scheme_context}

{conversation_context}

Please provide a comprehensive response that addresses both the disease/pest issue and available government schemes. Structure your response clearly with sections if needed.""")
            ])
            
            disease_context = self._format_documents_for_prompt(disease_docs) if disease_docs else "No disease information found."
            scheme_context = self._format_documents_for_prompt(scheme_docs) if scheme_docs else "No scheme information found."
            
            # Format conversation history if provided
            conv_context = ""
            if conversation_history:
                conv_context = "\n\nPrevious Conversation Context:\n"
                for i, msg in enumerate(conversation_history[-3:], 1):  # Last 3 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conv_context += f"{role.capitalize()}: {content}\n"
            
            prompt = prompt_template.format_messages(
                query=query,
                disease_context=disease_context,
                scheme_context=scheme_context,
                conversation_context=conv_context
            )
        else:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful agricultural assistant for farmers. Your task is to provide clear, actionable, and farmer-friendly responses based on the retrieved information from knowledge bases.

Guidelines:
- Use simple, clear language suitable for farmers
- Provide actionable steps and recommendations
- Cite information from sources when relevant (e.g., "According to [Source 1]...")
- Be specific and accurate
- If information is not available, say so clearly
- Focus on practical, field-ready advice
- Use numbered lists or bullet points for clarity when appropriate

Respond in a helpful, conversational tone."""),
                ("human", """User Query: {query}

Retrieved Context:
{context}

{conversation_context}

Please provide a comprehensive response based on the retrieved information.""")
            ])
            
            # Format conversation history if provided
            conv_context = ""
            if conversation_history:
                conv_context = "\n\nPrevious Conversation Context:\n"
                for i, msg in enumerate(conversation_history[-3:], 1):  # Last 3 messages
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    conv_context += f"{role.capitalize()}: {content}\n"
            
            prompt = prompt_template.format_messages(
                query=query,
                context=context,
                conversation_context=conv_context
            )
        
        # Generate response
        response = self.llm.invoke(prompt)
        
        return response.content
    
    def _format_documents_for_prompt(self, docs: List[Document]) -> str:
        """Format documents for prompt with citations including page numbers and confidence.
        
        Args:
            docs: List of Document objects with metadata (including confidence and page_number)
            
        Returns:
            Formatted string for prompt with citations
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


def main():
    """Test the response generator."""
    print("🚀 Response Generator Test")
    print("="*60)
    
    generator = ResponseGenerator()
    
    # Test case
    test_query = "What is citrus canker?"
    test_context = "[Source 1]\nCitrus canker is a bacterial disease caused by Xanthomonas citri pv. citri.\n"
    
    try:
        response = generator.generate_response(
            query=test_query,
            context=test_context,
            intent="disease"
        )
        
        print(f"\nQuery: {test_query}")
        print(f"\nContext: {test_context}")
        print(f"\nGenerated Response:")
        print(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Response Generator Test Complete")


if __name__ == "__main__":
    main()

