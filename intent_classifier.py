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

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_MODEL = "gpt-4"  # Using GPT-4 for intent classification
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

Classify the following query and provide your reasoning.
"""


class IntentClassifier:
    """Intent classifier using OpenAI GPT-4 with structured output."""
    
    def __init__(self, model_name: str = OPENAI_MODEL, temperature: float = 0.0):
        """Initialize the intent classifier.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4)
            temperature: Temperature for generation (default: 0.0 for deterministic classification)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
        
        # Create structured output chain
        # Using function_calling method (GPT-4 doesn't support json_schema method)
        self.structured_llm = self.llm.with_structured_output(
            IntentClassification,
            method="function_calling"
        )
    
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
        
        # Get structured output
        result = self.structured_llm.invoke(messages)
        
        # Convert Pydantic model to dict
        return {
            "query": result.query,
            "intent": result.intent,
            "confidence": result.confidence,
            "reasoning": result.reasoning
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

