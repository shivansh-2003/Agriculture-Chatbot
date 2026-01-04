# Agriculture Chatbot API Documentation

## Overview

The Agriculture Chatbot API is a FastAPI-based REST API that provides intelligent responses to farmer queries about citrus diseases and government agricultural schemes. The API uses LangChain, LangGraph, and OpenAI to process queries through an agentic workflow that includes intent classification, query routing, and retrieval-augmented generation (RAG).

**Base URL**: `http://localhost:8000` (development) or your deployed URL

**API Version**: 1.0.0

---

## Table of Contents

- [Getting Started](#getting-started)
- [Endpoints](#endpoints)
  - [Root Endpoint](#root-endpoint)
  - [Health Check](#health-check)
  - [Query Endpoint](#query-endpoint)
- [Request/Response Models](#requestresponse-models)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Authentication](#authentication)
- [Rate Limiting](#rate-limiting)

---

## Getting Started

### Prerequisites

- Python 3.9+
- All environment variables configured (see `.env.example`)
- Vector database (Pinecone) populated with documents

### Running the API

```bash
# Using Python directly
python api.py

# Or using uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Endpoints

### Root Endpoint

**GET** `/`

Returns API information and available endpoints.

#### Response

```json
{
  "message": "Agriculture Chatbot API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "endpoints": {
    "POST /query": "Submit a farmer query",
    "GET /health": "Health check"
  }
}
```

#### Example Request

```bash
curl http://localhost:8000/
```

---

### Health Check

**GET** `/health`

Checks the health status of the API and verifies that the chatbot is initialized and ready.

#### Response Model

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Health status: `"healthy"` or `"unhealthy"` |
| `message` | string | Status message |

#### Response Examples

**Healthy:**

```json
{
  "status": "healthy",
  "message": "API is operational and chatbot is ready"
}
```

**Unhealthy:**

```json
{
  "status": "unhealthy",
  "message": "Error: <error details>"
}
```

#### Example Request

```bash
curl http://localhost:8000/health
```

---

### Query Endpoint

**POST** `/query`

Processes a farmer's query about citrus diseases or government schemes. This is the main endpoint that:

1. Classifies the query intent (disease/scheme/hybrid)
2. Routes to appropriate knowledge bases
3. Retrieves relevant information using semantic search
4. Generates a farmer-friendly response with citations

#### Request Model

| Field | Type | Required | Description | Constraints |
|-------|------|----------|-------------|-------------|
| `question` | string | Yes | The farmer's question about citrus diseases or government schemes | Min: 1, Max: 1000 characters |
| `thread_id` | string | No | Optional thread ID for conversation history. If not provided, a new conversation is started. | - |

#### Request Example

```json
{
  "question": "What government schemes are available for citrus farmers in Andhra Pradesh?",
  "thread_id": "user-123-session-1"
}
```

#### Response Model

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the query was processed successfully |
| `intent` | string | Detected intent: `"disease"`, `"scheme"`, or `"hybrid"` |
| `answer` | string | The generated response to the farmer's query |
| `confidence` | float (optional) | Confidence score for intent classification (0-1) |
| `reasoning` | string (optional) | Reasoning for intent classification |
| `num_disease_docs` | integer (optional) | Number of disease documents retrieved |
| `num_scheme_docs` | integer (optional) | Number of scheme documents retrieved |

#### Response Example

```json
{
  "success": true,
  "intent": "scheme",
  "answer": "Several government schemes are available for citrus farmers in Andhra Pradesh:\n\n1. Pradhan Mantri Krishi Sinchai Yojana (PMKSY)...",
  "confidence": 0.95,
  "reasoning": "Query focuses on government schemes and subsidies",
  "num_disease_docs": 0,
  "num_scheme_docs": 3
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success - Query processed successfully |
| 400 | Bad Request - Invalid request format or empty question |
| 500 | Internal Server Error - Error during query processing |

#### Example Request

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What government schemes are available for citrus farmers in Andhra Pradesh?"
  }'
```

---

## Request/Response Models

### QueryRequest

```python
class QueryRequest(BaseModel):
    question: str  # Required, 1-1000 characters
    thread_id: Optional[str] = None  # Optional, for conversation history
```

### QueryResponse

```python
class QueryResponse(BaseModel):
    success: bool
    intent: str  # "disease", "scheme", or "hybrid"
    answer: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    num_disease_docs: Optional[int] = None
    num_scheme_docs: Optional[int] = None
```

### HealthResponse

```python
class HealthResponse(BaseModel):
    status: str  # "healthy" or "unhealthy"
    message: str
```

---

## Examples

### Disease Intent Query

**Request:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "My citrus leaves are showing yellow blotchy patches. What could this be?"
  }'
```

**Response:**

```json
{
  "success": true,
  "intent": "disease",
  "answer": "The yellow blotchy patches on your citrus leaves could indicate Huanglongbing (HLB) or Citrus Greening disease. This is a serious bacterial disease transmitted by the Asian citrus psyllid...",
  "confidence": 0.92,
  "reasoning": "Query is about disease symptoms and identification",
  "num_disease_docs": 3,
  "num_scheme_docs": 0
}
```

### Scheme Intent Query

**Request:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What government schemes are available for citrus farmers in Andhra Pradesh?"
  }'
```

**Response:**

```json
{
  "success": true,
  "intent": "scheme",
  "answer": "Several government schemes are available for citrus farmers in Andhra Pradesh:\n\n1. Pradhan Mantri Krishi Sinchai Yojana (PMKSY)...",
  "confidence": 0.95,
  "reasoning": "Query focuses on government schemes and subsidies",
  "num_disease_docs": 0,
  "num_scheme_docs": 3
}
```

### Hybrid Intent Query

**Request:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can I get government support for setting up drip irrigation to prevent root diseases?"
  }'
```

**Response:**

```json
{
  "success": true,
  "intent": "hybrid",
  "answer": "Yes, drip irrigation helps prevent root diseases and government support is available:\n\nDISEASE PREVENTION BENEFITS:\nDrip irrigation significantly reduces Phytophthora foot rot...\n\nGOVERNMENT SUBSIDY:\nPradhan Mantri Krishi Sinchai Yojana (PMKSY)...",
  "confidence": 0.88,
  "reasoning": "Query combines disease prevention and government schemes",
  "num_disease_docs": 2,
  "num_scheme_docs": 3
}
```

### Conversation History (Multi-turn)

**Request 1:**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is citrus canker?",
    "thread_id": "user-123"
  }'
```

**Request 2 (with context):**

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What schemes help with it?",
    "thread_id": "user-123"
  }'
```

The second request will have context from the first query, allowing for natural follow-up conversations.

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message description"
}
```

### Common Error Codes

| Status Code | Error Type | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | Bad Request | Invalid request format, empty question, or validation error | Check request body format and ensure question is not empty |
| 500 | Internal Server Error | Error during query processing (e.g., API key issues, vector DB connection errors) | Check server logs, verify environment variables, ensure vector DB is accessible |

### Error Examples

**Empty Question:**

```json
{
  "detail": "Question cannot be empty"
}
```

**Invalid Request:**

```json
{
  "detail": "Invalid request: <specific error>"
}
```

**Internal Server Error:**

```json
{
  "detail": "An error occurred while processing your query: <error details>"
}
```

---

## Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:

- API key authentication
- OAuth 2.0 / JWT tokens
- Rate limiting per API key

---

## Rate Limiting

Rate limiting is not currently implemented. For production deployments, consider:

- Request rate limits per IP/API key
- Quota-based limiting (requests per day/month)
- Throttling to prevent abuse

Recommended limits:
- **Free tier**: 100 requests/hour
- **Paid tier**: 1000 requests/hour

---

## CORS

The API includes CORS middleware configured to allow all origins (`*`). **For production, restrict this to specific domains:**

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Performance Considerations

### Response Times

- **Typical response time**: 10-40 seconds
- **Intent classification**: 5-10 seconds
- **Query rewriting** (hybrid only): 3-5 seconds
- **Document retrieval**: 2-5 seconds
- **Response generation**: 10-25 seconds

### Optimization Tips

1. **Use conversation history**: Reuse `thread_id` for multi-turn conversations
2. **Keep questions concise**: Shorter questions process faster
3. **Cache common queries**: Implement caching for frequently asked questions

---

## Testing

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is citrus canker?"}'
```

### Using Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Query endpoint
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "What is citrus canker?",
        "thread_id": "test-123"
    }
)
print(response.json())
```

### Using the Interactive Documentation

Visit `http://localhost:8000/docs` to test endpoints directly in your browser.

---

## Support

For issues, questions, or contributions:

- **GitHub Issues**: [Repository URL]
- **Email**: support@alumnx.com
- **Documentation**: See `README.md` for setup and architecture details

---

## Changelog

### Version 1.0.0

- Initial release
- Intent classification (disease/scheme/hybrid)
- Query rewriting for hybrid intents
- RAG-based response generation
- Conversation history support
- Health check endpoint

---

**Last Updated**: January 2026

