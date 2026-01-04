#!/usr/bin/env python3
"""
Phase 2: Query Rewriter for Hybrid Intent
Query Rewriter for Agriculture Chatbot

This module rewrites hybrid intent queries into two separate queries:
1. Disease-focused query → for Citrus Pests & Diseases KB
2. Scheme-focused query → for Government Schemes KB
"""

import os
import json
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"  # Using OpenAI for query rewriting
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RewrittenQueries(BaseModel):
    """Structured output schema for rewritten queries."""
    model_config = ConfigDict(strict=True)
    
    disease_query: str = Field(
        description="Rewritten query focused on disease/pest aspects for the Citrus Pests & Diseases knowledge base"
    )
    scheme_query: str = Field(
        description="Rewritten query focused on government schemes/subsidies for the Government Schemes knowledge base"
    )
    reasoning: str = Field(
        description="Brief explanation of how the original query was split into two focused queries"
    )


# Prompt template for query rewriting
QUERY_REWRITE_PROMPT = """You are an expert query rewriter for an agricultural chatbot that helps farmers with citrus crop diseases and government agricultural schemes.

Your task is to rewrite a hybrid intent query (one that requires information from both knowledge bases) into TWO separate, focused queries:

1. **Disease Query**: Extract and rewrite the disease/pest-related aspects of the query for the Citrus Pests & Diseases knowledge base
   - Focus on: symptoms, disease/pest identification, treatment, prevention, management
   - Remove scheme/subsidy/financial assistance references
   - Make it a clear, standalone query about the agricultural problem

2. **Scheme Query**: Extract and rewrite the scheme/subsidy-related aspects of the query for the Government Schemes knowledge base
   - Focus on: government programs, subsidies, financial assistance, eligibility, application processes
   - Remove specific disease/pest treatment details
   - Make it a clear, standalone query about available support/schemes

**Examples:**

**Example 1:**
Original Query: "Can I get government support for setting up drip irrigation to prevent root diseases?"
- Disease Query: "How does drip irrigation help prevent root diseases in citrus?"
- Scheme Query: "What government subsidies are available for setting up drip irrigation systems?"

**Example 2:**
Original Query: "What schemes can help me manage Citrus Canker?"
- Disease Query: "What is Citrus Canker and how is it managed?"
- Scheme Query: "What government schemes provide financial assistance for disease management in citrus farming?"

**Example 3:**
Original Query: "Tell me about Citrus Canker and available government schemes"
- Disease Query: "What is Citrus Canker, its symptoms, and management?"
- Scheme Query: "What government schemes are available for citrus farmers?"

**Guidelines:**
- Each rewritten query should be self-contained and clear
- Disease queries should focus on agricultural/technical aspects
- Scheme queries should focus on financial/government support aspects
- Preserve the core intent and meaning of the original query
- Make queries specific enough to retrieve relevant information from each knowledge base

**IMPORTANT: You MUST respond with valid JSON only, in this exact format:**
{
  "disease_query": "the disease-focused query",
  "scheme_query": "the scheme-focused query",
  "reasoning": "brief explanation of how the query was split"
}

Rewrite the following hybrid query into two focused queries.
"""


class QueryRewriter:
    """Query rewriter for hybrid intent queries using OpenAI LLM with structured output."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.0):
        """Initialize the query rewriter.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4o-mini)
            temperature: Temperature for generation (default: 0.0 for deterministic rewriting)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=OPENAI_API_KEY
        )
        
        # Create JSON parser as fallback
        self.json_parser = JsonOutputParser(pydantic_object=RewrittenQueries)
        
        # Try structured output with OpenAI (uses function calling)
        try:
            self.structured_llm = self.llm.with_structured_output(
                RewrittenQueries,
                method="function_calling"
            )
        except Exception:
            # Fallback to regular LLM with JSON parser
            self.structured_llm = None
    
    @traceable(name="query_rewriter.rewrite")
    def rewrite(self, query: str) -> dict:
        """Rewrite a hybrid intent query into two focused queries.
        
        Args:
            query: The original hybrid intent query
            
        Returns:
            Dictionary with keys: disease_query, scheme_query, reasoning
        """
        # Create the prompt with the query
        messages = [
            SystemMessage(content=QUERY_REWRITE_PROMPT),
            HumanMessage(content=f"Original Hybrid Query: {query}")
        ]
        
        # Try structured output first
        try:
            if self.structured_llm:
                result = self.structured_llm.invoke(messages)
                # Convert Pydantic model to dict
                return {
                    "disease_query": result.disease_query,
                    "scheme_query": result.scheme_query,
                    "reasoning": result.reasoning
                }
        except Exception as e:
            print(f"⚠️  Structured output failed: {e}, trying JSON parser fallback...")
        
        # Fallback: Use regular LLM with JSON parser
        try:
            response = self.llm.invoke(messages)
            # Parse JSON from response
            result_dict = self.json_parser.parse(response.content)
            # Validate and return
            return {
                "disease_query": result_dict.get("disease_query", query),
                "scheme_query": result_dict.get("scheme_query", query),
                "reasoning": result_dict.get("reasoning", "Query rewritten into two focused queries")
            }
        except Exception as e:
            print(f"⚠️  JSON parsing failed: {e}, using simple fallback...")
            # Ultimate fallback: simple split based on keywords
            query_lower = query.lower()
            # Extract disease-related parts
            disease_keywords = ["disease", "pest", "canker", "treatment", "control", "symptom", "prevent", "manage"]
            scheme_keywords = ["scheme", "subsidy", "government", "financial", "assistance", "support", "program"]
            
            disease_query = query if any(kw in query_lower for kw in disease_keywords) else f"What diseases or pests are mentioned in: {query}"
            scheme_query = query if any(kw in query_lower for kw in scheme_keywords) else f"What government schemes are available for: {query}"
            
            return {
                "disease_query": disease_query,
                "scheme_query": scheme_query,
                "reasoning": "Fallback rewriting based on keyword extraction"
            }


def main():
    """Test the query rewriter with example hybrid queries."""
    print("🚀 Query Rewriter Test")
    print("="*60)
    
    # Initialize rewriter
    rewriter = QueryRewriter()
    
    # Test queries (hybrid intent examples)
    test_queries = [
        "Can I get government support for setting up drip irrigation to prevent root diseases?",
        "What schemes can help me manage Citrus Canker?",
        "Tell me about Citrus Canker and available government schemes",
        "Are there any government programs for organic pest control in citrus?",
    ]
    
    print("\nTesting Query Rewriting for Hybrid Intent:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Original Query {i}: {query}")
        print("-"*60)
        
        try:
            result = rewriter.rewrite(query)
            print(f"\nDisease Query: {result['disease_query']}")
            print(f"\nScheme Query: {result['scheme_query']}")
            print(f"\nReasoning: {result['reasoning']}")
            print(f"\nJSON Output:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Query Rewriter Test Complete")


if __name__ == "__main__":
    main()

