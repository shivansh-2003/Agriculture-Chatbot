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
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from langsmith import traceable

from intent_classifier import IntentClassifier
from query_rewriter import QueryRewriter
from retrieve import RetrievalSystem

# Load environment variables
load_dotenv()

# LangSmith tracing is automatically enabled when LANGSMITH_TRACING=true is set
# Check if tracing is enabled

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if LANGSMITH_TRACING:
    print("✅ LangSmith tracing enabled")
    print(f"   LangSmith API Key: {'Set' if os.getenv('LANGSMITH_API_KEY') else 'Not Set'}")
    print(f"   Workspace ID: {os.getenv('LANGSMITH_WORKSPACE_ID', 'Default')}")
else:
    print("⚠️  LangSmith tracing is disabled. Set LANGSMITH_TRACING=true to enable.")


class AgentState(TypedDict):
    """State schema for the LangGraph workflow."""
    messages: Annotated[list, add_messages]  # Conversation history with reducer
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
        
        # Initialize checkpointer for conversation history
        self.checkpointer = InMemorySaver()
        
        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
    
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
    
    @traceable(name="classify_intent_node")
    def _classify_intent_node(self, state: AgentState) -> AgentState:
        """Node: Classify the user query intent.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with intent classification
        """
        # Get user query from messages or state
        messages = state.get("messages", [])
        if messages:
            user_query = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])
        else:
            user_query = state.get("user_query", "")
        
        print(f"\n[Classify Intent Node] Processing query: {user_query}")
        
        # Classify intent
        classification = self.intent_classifier.classify(user_query)
        
        print(f"[Classify Intent Node] Intent: {classification['intent']}, Confidence: {classification['confidence']:.2f}")
        
        # Prepare state update
        state_update = {
            "user_query": user_query,
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
    
    @traceable(name="rewrite_query_node")
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
    
    @traceable(name="retrieve_and_generate_node")
    def _retrieve_and_generate_node(self, state: AgentState) -> AgentState:
        """Node: Retrieve relevant documents and generate response.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with response (Document objects are not stored in state due to serialization limits)
        """
        user_query = state["user_query"]
        intent = state.get("intent")
        disease_query = state.get("disease_query")
        scheme_query = state.get("scheme_query")
        
        # Get conversation history from messages
        messages = state.get("messages", [])
        conversation_history = []
        for msg in messages[:-1]:  # Exclude current message
            if isinstance(msg, (HumanMessage, AIMessage)):
                conversation_history.append({
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content if hasattr(msg, 'content') else str(msg)
                })
        
        print(f"\n[Retrieve & Generate Node] Intent: {intent}")
        print(f"[Retrieve & Generate Node] Processing query: {user_query}")
        
        # Retrieve and generate response
        result = self.retrieval_system.retrieve_and_generate(
            query=user_query,
            intent=intent,
            disease_query=disease_query,
            scheme_query=scheme_query,
            conversation_history=conversation_history if conversation_history else None
        )
        
        print(f"[Retrieve & Generate Node] Retrieved {result['num_disease_docs']} disease docs, {result['num_scheme_docs']} scheme docs")
        if result.get('response'):
            print(f"[Retrieve & Generate Node] Generated response ({len(result['response'])} chars)")
        
        # Store documents temporarily (they can't be serialized in checkpointer state)
        # Store them in the graph instance for retrieval after invoke
        self._temp_retrieval_result = {
            "disease_docs": result.get("disease_docs"),
            "scheme_docs": result.get("scheme_docs"),
            "context": result.get("context")
        }
        
        # Add AI response to messages
        response_text = result.get("response", "")
        new_messages = []
        if response_text:
            new_messages.append(AIMessage(content=response_text))
        
        # Update state - Don't include Document objects (they can't be serialized)
        # Store only serializable fields
        return {
            "messages": new_messages,
            "response": response_text,
            "num_disease_docs": result["num_disease_docs"],
            "num_scheme_docs": result["num_scheme_docs"]
        }
    
    def process_query(self, user_query: str, thread_id: str = "default") -> dict:
        """Process a user query through the graph with conversation history.
        
        Args:
            user_query: The farmer's query/question
            thread_id: Thread ID for conversation history (default: "default")
            
        Returns:
            Dictionary containing the complete processing result
        """
        # Create human message
        human_message = HumanMessage(content=user_query)
        
        # Initial state with message
        initial_state = {
            "messages": [human_message],
            "user_query": user_query,
            "intent": None,
            "confidence": None,
            "reasoning": None,
            "disease_query": None,
            "scheme_query": None,
            "rewrite_reasoning": None,
            "response": None,
            "disease_docs": None,
            "scheme_docs": None,
            "num_disease_docs": None,
            "num_scheme_docs": None
        }
        
        # Run the graph with thread_id for conversation history
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke(initial_state, config)
        
        # Add Document objects from temporary storage (not stored in checkpointer state)
        result["disease_docs"] = self._temp_retrieval_result.get("disease_docs")
        result["scheme_docs"] = self._temp_retrieval_result.get("scheme_docs")
        result["context"] = self._temp_retrieval_result.get("context")
        
        # Clear temporary storage
        self._temp_retrieval_result = {}
        
        return result


def main():
    """Test the LangGraph workflow with example queries."""
    print("🚀 Agriculture Chatbot - LangGraph Workflow Test")
    print("="*80)
    
    # Check LangSmith configuration
    if LANGSMITH_TRACING:
        print("\n📊 LangSmith Observability:")
        print(f"   ✅ Tracing: Enabled")
        print(f"   🔑 API Key: {'✅ Set' if os.getenv('LANGSMITH_API_KEY') else '❌ Not Set'}")
        workspace_id = os.getenv('LANGSMITH_WORKSPACE_ID', 'Default')
        print(f"   🏢 Workspace: {workspace_id}")
        print(f"   📈 View traces at: https://smith.langchain.com")
        print()
    else:
        print("\n⚠️  LangSmith tracing is disabled. Set LANGSMITH_TRACING=true to enable.")
        print()
    
    # Initialize the graph
    chatbot = AgricultureChatbotGraph()
    
    # Print the graph structure
    chatbot.print_graph()
    
    # Test queries covering all three intent types
    test_queries = [
        
        # Hybrid Intent
        "Can I get government support for setting up drip irrigation to prevent root diseases?",
       
    ]
    
    print("\nTesting LangGraph Workflow:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Test Query {i}: {query}")
        print(f"{'='*80}")
        
        try:
            # Use same thread_id for all queries in test to show conversation history
            thread_id = "test-session-1"
            result = chatbot.process_query(query, thread_id=thread_id)
            
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
            
            # Show retrieved document content
            if result.get('disease_docs'):
                print(f"\n  📄 Disease Documents Content:")
                for i, doc in enumerate(result['disease_docs'][:3], 1):  # Show first 3
                    content = doc.page_content[:300] if len(doc.page_content) > 300 else doc.page_content
                    print(f"\n    [Disease Doc {i}]")
                    print(f"    {content}{'...' if len(doc.page_content) > 300 else ''}")
                    if doc.metadata:
                        print(f"    Metadata: {doc.metadata.get('source', 'Unknown')}")
            
            if result.get('scheme_docs'):
                print(f"\n  📄 Scheme Documents Content:")
                for i, doc in enumerate(result['scheme_docs'][:3], 1):  # Show first 3
                    content = doc.page_content[:300] if len(doc.page_content) > 300 else doc.page_content
                    print(f"\n    [Scheme Doc {i}]")
                    print(f"    {content}{'...' if len(doc.page_content) > 300 else ''}")
                    if doc.metadata:
                        print(f"    Metadata: {doc.metadata.get('source', 'Unknown')}")
            
            # Show formatted context
            if result.get('context'):
                print(f"\n  📝 Formatted Context (from vector DB):")
                context = result['context']
                if len(context) > 1000:
                    print(f"    {context[:1000]}...")
                    print(f"    ... (truncated, total length: {len(context)} chars)")
                else:
                    print(f"    {context}")
            
            print(f"\n  💬 Generated Response:")
            response_text = result.get('response', 'N/A')
            if response_text and response_text != 'N/A':
                # Print full response
                print(f"    {response_text}")
            else:
                print(f"    (Response generation disabled - see retrieved documents above)")
            
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

