#!/usr/bin/env python3
"""
Phase 2: Main LangGraph Workflow
Agriculture Chatbot - Intent Classification and Query Routing

This module implements a LangGraph workflow that:
1. Classifies user queries into disease, scheme, or hybrid intent
2. Rewrites hybrid queries into two focused queries
3. Routes queries to appropriate knowledge bases
"""

import os
import json
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from intent_classifier import IntentClassifier
from query_rewriter import QueryRewriter
from retrieve import RetrievalSystem

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State schema for the LangGraph workflow."""
    user_query: str
    intent: Literal["disease", "scheme", "hybrid"] | None
    confidence: float | None
    reasoning: str | None
    disease_query: str | None
    scheme_query: str | None
    rewrite_reasoning: str | None
    response: str | None
    disease_docs: list | None
    scheme_docs: list | None
    num_disease_docs: int | None
    num_scheme_docs: int | None


class AgricultureChatbotGraph:
    """LangGraph workflow for agriculture chatbot intent classification and query routing."""
    
    def __init__(self):
        """Initialize the chatbot graph with intent classifier, query rewriter, and retrieval system."""
        self.intent_classifier = IntentClassifier()
        self.query_rewriter = QueryRewriter()
        self.retrieval_system = RetrievalSystem()
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()
    
    def print_graph(self):
        """Print the LangGraph workflow structure."""
        print("\n" + "="*80)
        print("📊 LangGraph Workflow Structure")
        print("="*80)
        print("\nGraph Nodes and Edges:\n")
        try:
            # Get graph representation
            graph_str = self.app.get_graph().print_ascii()
            print(graph_str)
        except Exception as e:
            # Fallback: manually describe the graph
            print("Workflow Structure:")
            print("  Entry Point: classify_intent")
            print("\n  Nodes:")
            print("    1. classify_intent - Classifies user query intent")
            print("    2. rewrite_query - Rewrites hybrid queries into two queries")
            print("    3. retrieve_and_generate - Retrieves documents and generates response")
            print("\n  Edges:")
            print("    classify_intent → [conditional routing]")
            print("      - If intent='disease' → retrieve_and_generate → END")
            print("      - If intent='scheme' → retrieve_and_generate → END")
            print("      - If intent='hybrid' → rewrite_query → retrieve_and_generate → END")
            print(f"\n  Error getting ASCII representation: {e}")
        
        print("\n" + "="*80 + "\n")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph.
        
        Returns:
            Compiled StateGraph
        """
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify_intent", self._classify_intent_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        workflow.add_node("retrieve_and_generate", self._retrieve_and_generate_node)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional edges based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "disease": "retrieve_and_generate",
                "scheme": "retrieve_and_generate",
                "hybrid": "rewrite_query"
            }
        )
        
        # Rewrite query node goes to retrieve_and_generate
        workflow.add_edge("rewrite_query", "retrieve_and_generate")
        
        # Retrieve and generate node goes to END
        workflow.add_edge("retrieve_and_generate", END)
        
        return workflow
    
    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Node: Classify the user query intent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with intent classification
        """
        user_query = state["user_query"]
        
        print(f"\n[Classify Intent Node] Processing query: {user_query}")
        
        # Classify intent
        classification = self.intent_classifier.classify(user_query)
        
        print(f"[Classify Intent Node] Intent: {classification['intent']}, Confidence: {classification['confidence']:.2f}")
        
        # Prepare state update
        state_update = {
            "intent": classification["intent"],
            "confidence": classification["confidence"],
            "reasoning": classification["reasoning"]
        }
        
        # Populate query field based on intent (non-hybrid cases)
        if classification["intent"] == "disease":
            state_update["disease_query"] = user_query
        elif classification["intent"] == "scheme":
            state_update["scheme_query"] = user_query
        # For hybrid, we'll populate in rewrite_query_node
        
        return state_update
    
    def _rewrite_query_node(self, state: AgentState) -> AgentState:
        """Node: Rewrite hybrid query into two focused queries.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with rewritten queries
        """
        user_query = state["user_query"]
        
        print(f"\n[Rewrite Query Node] Rewriting hybrid query: {user_query}")
        
        # Rewrite query
        rewritten = self.query_rewriter.rewrite(user_query)
        
        print(f"[Rewrite Query Node] Disease Query: {rewritten['disease_query']}")
        print(f"[Rewrite Query Node] Scheme Query: {rewritten['scheme_query']}")
        
        # Update state
        return {
            "disease_query": rewritten["disease_query"],
            "scheme_query": rewritten["scheme_query"],
            "rewrite_reasoning": rewritten["reasoning"]
        }
    
    def _route_after_classification(self, state: AgentState) -> str:
        """Route to next node based on intent classification.
        
        Args:
            state: Current agent state
            
        Returns:
            Next node name ("disease", "scheme", or "hybrid")
        """
        intent = state.get("intent")
        
        if intent == "disease":
            return "disease"
        elif intent == "scheme":
            return "scheme"
        elif intent == "hybrid":
            return "hybrid"
        else:
            # Default to hybrid if intent is unclear
            return "hybrid"
    
    def _retrieve_and_generate_node(self, state: AgentState) -> AgentState:
        """Node: Retrieve relevant documents and generate response.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with response and retrieved documents
        """
        user_query = state["user_query"]
        intent = state.get("intent")
        disease_query = state.get("disease_query")
        scheme_query = state.get("scheme_query")
        
        print(f"\n[Retrieve & Generate Node] Intent: {intent}")
        print(f"[Retrieve & Generate Node] Processing query: {user_query}")
        
        # Retrieve and generate response
        result = self.retrieval_system.retrieve_and_generate(
            query=user_query,
            intent=intent,
            disease_query=disease_query,
            scheme_query=scheme_query
        )
        
        print(f"[Retrieve & Generate Node] Retrieved {result['num_disease_docs']} disease docs, {result['num_scheme_docs']} scheme docs")
        print(f"[Retrieve & Generate Node] Generated response ({len(result['response'])} chars)")
        
        # Update state
        return {
            "response": result["response"],
            "disease_docs": result.get("disease_docs"),
            "scheme_docs": result.get("scheme_docs"),
            "num_disease_docs": result["num_disease_docs"],
            "num_scheme_docs": result["num_scheme_docs"]
        }
    
    def process_query(self, user_query: str) -> dict:
        """Process a user query through the graph.
        
        Args:
            user_query: The farmer's query/question
            
        Returns:
            Dictionary containing the complete processing result
        """
        # Initial state
        initial_state = AgentState(
            user_query=user_query,
            intent=None,
            confidence=None,
            reasoning=None,
            disease_query=None,
            scheme_query=None,
            rewrite_reasoning=None,
            response=None,
            disease_docs=None,
            scheme_docs=None,
            num_disease_docs=None,
            num_scheme_docs=None
        )
        
        # Run the graph
        result = self.app.invoke(initial_state)
        
        return result


def main():
    """Test the LangGraph workflow with example queries."""
    print("🚀 Agriculture Chatbot - LangGraph Workflow Test")
    print("="*80)
    
    # Initialize the graph
    chatbot = AgricultureChatbotGraph()
    
    # Print the graph structure
    chatbot.print_graph()
    
    # Test queries covering all three intent types
    test_queries = [
        # Disease Intent
        "My citrus leaves are showing yellow blotchy patches. What could this be?",
        
        # Scheme Intent
        "What government schemes are available for citrus farmers in Andhra Pradesh?",
        
        # Hybrid Intent
        "Can I get government support for setting up drip irrigation to prevent root diseases?",
        "What schemes can help me manage Citrus Canker?",
    ]
    
    print("\nTesting LangGraph Workflow:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*80}")
        
        try:
            result = chatbot.process_query(query)
            
            print(f"\n📊 Final Result:")
            print(f"  Intent: {result.get('intent')}")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
            
            if result.get("intent") == "hybrid":
                print(f"\n  📝 Rewritten Queries:")
                print(f"    Disease Query: {result.get('disease_query', 'N/A')}")
                print(f"    Scheme Query: {result.get('scheme_query', 'N/A')}")
                print(f"    Rewrite Reasoning: {result.get('rewrite_reasoning', 'N/A')}")
            
            print(f"\n  📚 Retrieved Documents:")
            print(f"    Disease Docs: {result.get('num_disease_docs', 0)}")
            print(f"    Scheme Docs: {result.get('num_scheme_docs', 0)}")
            
            print(f"\n  💬 Generated Response:")
            response_text = result.get('response', 'N/A')
            if response_text and response_text != 'N/A':
                # Print first 500 chars, then ...
                if len(response_text) > 500:
                    print(f"    {response_text[:500]}...")
                else:
                    print(f"    {response_text}")
            
            print(f"\n  📄 Complete JSON Output:")
            # Convert documents to summary for JSON output (documents can be large)
            json_result = {
                k: v for k, v in result.items() 
                if k not in ['disease_docs', 'scheme_docs']
            }
            if result.get('disease_docs'):
                json_result['disease_docs_count'] = len(result['disease_docs'])
            if result.get('scheme_docs'):
                json_result['scheme_docs_count'] = len(result['scheme_docs'])
            print(json.dumps(json_result, indent=2, default=str))
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ LangGraph Workflow Test Complete")


if __name__ == "__main__":
    main()

