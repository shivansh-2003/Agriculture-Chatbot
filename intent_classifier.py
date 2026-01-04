#!/usr/bin/env python3
"""
Phase 2: Intent Detection & Agentic Routing
Intent Classifier for Agriculture Chatbot

This module classifies farmer queries into three categories:
1. Disease Intent → Route to Citrus Pests & Diseases KB
2. Scheme Intent → Route to Government Schemes KB
3. Hybrid Intent → Route to BOTH knowledge bases
"""

import os
import json
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable

# Load environment variables
load_dotenv()

# Ollama Configuration
OLLAMA_MODEL = "gpt-oss:20b"  # Using Ollama for intent classification

# Intent types
IntentType = Literal["disease", "scheme", "hybrid"]


class IntentClassification(BaseModel):
    """Structured output schema for intent classification."""
    model_config = ConfigDict(strict=True)
    
    query: str = Field(description="The original user query")
    intent: IntentType = Field(
        description="The classified intent: 'disease' for disease/pest queries, 'scheme' for government scheme queries, or 'hybrid' for queries requiring both knowledge bases"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )


# Prompt template with examples from the Agriculture Chatbot PDF
INTENT_CLASSIFICATION_PROMPT = """You are an expert intent classifier for an agricultural chatbot that helps farmers with citrus crop diseases and government agricultural schemes.

Your task is to classify farmer queries into one of three categories:

1. **Disease Intent** (classify as "disease"): 
   - Queries about disease symptoms and identification
   - Pest problems and infestations
   - Treatment and prevention methods
   - Nutritional deficiencies
   - Plant health issues
   
2. **Scheme Intent** (classify as "scheme"):
   - Queries about government subsidies and financial assistance
   - Agricultural support programs
   - Eligibility criteria for schemes
   - Application processes
   - Available benefits for farmers

3. **Hybrid Intent** (classify as "hybrid"):
   - Financial support for disease management
   - Schemes that help with specific pest control
   - Government assistance combined with agricultural problems
   - Any query connecting schemes with diseases/pests

**Examples:**

**Disease Intent Examples:**
- "My citrus leaves are showing yellow blotchy patches. What could this be?" → disease
- "How do I prevent Citrus Canker in my orchard?" → disease
- "What treatment should I use for whitefly infestation on my citrus trees?" → disease
- "My citrus trees have yellow leaves with green veins. What's wrong?" → disease
- "How to control citrus psyllid?" → disease

**Scheme Intent Examples:**
- "What government schemes are available for citrus farmers in Andhra Pradesh?" → scheme
- "Are there any subsidies for setting up drip irrigation in my citrus farm?" → scheme
- "How can I get financial help to start organic citrus farming?" → scheme
- "What are the eligibility criteria for PMKSY scheme?" → scheme
- "Tell me about subsidies for farm equipment" → scheme

**Hybrid Intent Examples:**
- "Can I get government support for setting up drip irrigation to prevent root diseases?" → hybrid
- "What schemes can help me manage Citrus Canker?" → hybrid
- "Are there any government programs for organic pest control in citrus?" → hybrid
- "Tell me about Citrus Canker and available government schemes" → hybrid

**Classification Guidelines:**
- If the query mentions ONLY diseases, pests, symptoms, or treatments → classify as "disease"
- If the query mentions ONLY schemes, subsidies, financial assistance, or government programs → classify as "scheme"
- If the query mentions BOTH disease/pest issues AND schemes/subsidies → classify as "hybrid"
- If the query asks about financial help for managing a disease/pest → classify as "hybrid"
- When in doubt, consider the primary need: if the main question is about disease/pest, choose "disease"; if about schemes, choose "scheme"; if it bridges both, choose "hybrid"

**IMPORTANT: You MUST respond with valid JSON only, in this exact format:**
{
  "query": "the original user query",
  "intent": "disease" or "scheme" or "hybrid",
  "confidence": 0.0 to 1.0,
  "reasoning": "brief explanation"
}

Classify the following query and provide your reasoning.
"""


class IntentClassifier:
    """Intent classifier using Ollama LLM with structured output."""
    
    def __init__(self, model_name: str = OLLAMA_MODEL, temperature: float = 0.0):
        """Initialize the intent classifier.
        
        Args:
            model_name: Ollama model to use (default: gpt-oss:20b)
            temperature: Temperature for generation (default: 0.0 for deterministic classification)
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        
        # Create JSON parser as fallback
        self.json_parser = JsonOutputParser(pydantic_object=IntentClassification)
        
        # Try structured output, but we'll use JSON parser as fallback
        try:
            self.structured_llm = self.llm.with_structured_output(
                IntentClassification,
                method="json_schema"
            )
        except Exception:
            # Fallback to regular LLM with JSON parser
            self.structured_llm = None
    
    @traceable(name="intent_classifier.classify")
    def classify(self, query: str) -> dict:
        """Classify a query into disease, scheme, or hybrid intent.
        
        Args:
            query: The farmer's query/question
            
        Returns:
            Dictionary with keys: query, intent, confidence, reasoning
        """
        # Create the prompt with the query
        messages = [
            SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
            HumanMessage(content=f"Farmer Query: {query}")
        ]
        
        # Try structured output first
        try:
            if self.structured_llm:
                result = self.structured_llm.invoke(messages)
                # Convert Pydantic model to dict
                return {
                    "query": result.query,
                    "intent": result.intent,
                    "confidence": result.confidence,
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
                "query": result_dict.get("query", query),
                "intent": result_dict.get("intent", "disease"),  # Default to disease
                "confidence": float(result_dict.get("confidence", 0.8)),
                "reasoning": result_dict.get("reasoning", "Classification completed")
            }
        except Exception as e:
            print(f"⚠️  JSON parsing failed: {e}, using default classification...")
            # Ultimate fallback: simple keyword-based classification
            query_lower = query.lower()
            if any(word in query_lower for word in ["scheme", "subsidy", "government", "financial", "assistance", "program"]):
                if any(word in query_lower for word in ["disease", "pest", "canker", "treatment", "control", "symptom"]):
                    intent = "hybrid"
                else:
                    intent = "scheme"
            elif any(word in query_lower for word in ["disease", "pest", "canker", "treatment", "control", "symptom", "yellow", "leaf"]):
                intent = "disease"
            else:
                intent = "disease"  # Default
            
            return {
                "query": query,
                "intent": intent,
                "confidence": 0.7,
                "reasoning": f"Fallback classification based on keywords"
            }
    
    def classify_batch(self, queries: list[str]) -> list[dict]:
        """Classify multiple queries.
        
        Args:
            queries: List of farmer queries
            
        Returns:
            List of classification dictionaries
        """
        return [self.classify(query) for query in queries]


def main():
    """Test the intent classifier with example queries."""
    print("🚀 Intent Classifier Test")
    print("="*60)
    
    # Initialize classifier
    classifier = IntentClassifier()
    
    # Test queries from the PDF examples
    test_queries = [
        # Disease Intent examples
        "My citrus leaves are showing yellow blotchy patches. What could this be?",
        "How do I prevent Citrus Canker in my orchard?",
        "What treatment should I use for whitefly infestation on my citrus trees?",
        
        # Scheme Intent examples
        "What government schemes are available for citrus farmers in Andhra Pradesh?",
        "Are there any subsidies for setting up drip irrigation in my citrus farm?",
        "How can I get financial help to start organic citrus farming?",
        
        # Hybrid Intent examples
        "Can I get government support for setting up drip irrigation to prevent root diseases?",
        "What schemes can help me manage Citrus Canker?",
        "Tell me about Citrus Canker and available government schemes",
    ]
    
    print("\nTesting Intent Classification:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print("-"*60)
        
        try:
            result = classifier.classify(query)
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"\nJSON Output:")
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("✅ Intent Classification Test Complete")


if __name__ == "__main__":
    main()

