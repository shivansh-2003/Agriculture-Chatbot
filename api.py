#!/usr/bin/env python3
"""
Phase 4: FastAPI Backend Development
Agriculture Chatbot API

This module provides FastAPI endpoints for the Agriculture Chatbot:
- POST /query - Main endpoint for farmer queries
- GET /health - Health check endpoint
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from main import AgricultureChatbotGraph

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Agriculture Chatbot API",
    description="An intelligent backend application that helps farmers get accurate information about citrus diseases and government agricultural schemes through a conversational interface.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the chatbot graph (singleton)
chatbot_graph = None

def get_chatbot():
    """Get or initialize the chatbot graph instance."""
    global chatbot_graph
    if chatbot_graph is None:
        chatbot_graph = AgricultureChatbotGraph()
    return chatbot_graph


# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(
        ...,
        description="The farmer's question about citrus diseases or government schemes",
        min_length=1,
        max_length=1000,
        example="What government schemes are available for citrus farmers in Andhra Pradesh?"
    )
    thread_id: Optional[str] = Field(
        None,
        description="Optional thread ID for conversation history. If not provided, a new conversation is started.",
        example="user-123-session-1"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    success: bool = Field(..., description="Whether the query was processed successfully")
    intent: str = Field(..., description="Detected intent: 'disease', 'scheme', or 'hybrid'")
    answer: str = Field(..., description="The generated response to the farmer's query")
    confidence: Optional[float] = Field(None, description="Confidence score for intent classification (0-1)")
    reasoning: Optional[str] = Field(None, description="Reasoning for intent classification")
    num_disease_docs: Optional[int] = Field(None, description="Number of disease documents retrieved")
    num_scheme_docs: Optional[int] = Field(None, description="Number of scheme documents retrieved")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Health status")
    message: str = Field(..., description="Health check message")


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Agriculture Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /query": "Submit a farmer query",
            "GET /health": "Health check"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Try to get chatbot instance (will initialize if needed)
        chatbot = get_chatbot()
        return {
            "status": "healthy",
            "message": "API is operational and chatbot is ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Error: {str(e)}"
        }


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """
    Process a farmer's query about citrus diseases or government schemes.
    
    This endpoint:
    1. Classifies the query intent (disease/scheme/hybrid)
    2. Routes to appropriate knowledge bases
    3. Retrieves relevant information
    4. Generates a farmer-friendly response
    
    **Example Requests:**
    
    **Disease Intent:**
    ```json
    {
        "question": "My citrus leaves are showing yellow blotchy patches. What could this be?"
    }
    ```
    
    **Scheme Intent:**
    ```json
    {
        "question": "What government schemes are available for citrus farmers in Andhra Pradesh?"
    }
    ```
    
    **Hybrid Intent:**
    ```json
    {
        "question": "Can I get government support for setting up drip irrigation to prevent root diseases?"
    }
    ```
    """
    try:
        # Validate input
        if not request.question or not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )
        
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Use provided thread_id or generate a default one
        thread_id = request.thread_id or "default"
        
        # Process the query through the LangGraph workflow
        result = chatbot.process_query(
            user_query=request.question,
            thread_id=thread_id
        )
        
        # Extract response
        response_text = result.get("response", "")
        intent = result.get("intent", "disease")
        confidence = result.get("confidence")
        reasoning = result.get("reasoning")
        num_disease_docs = result.get("num_disease_docs", 0)
        num_scheme_docs = result.get("num_scheme_docs", 0)
        
        # Handle case where response generation failed
        if not response_text:
            response_text = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
        
        # Return formatted response
        return QueryResponse(
            success=True,
            intent=intent,
            answer=response_text,
            confidence=confidence,
            reasoning=reasoning,
            num_disease_docs=num_disease_docs,
            num_scheme_docs=num_scheme_docs
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        # Log the error (in production, use proper logging)
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Starting Agriculture Chatbot API on http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"❤️  Health Check: http://{host}:{port}/health")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )

