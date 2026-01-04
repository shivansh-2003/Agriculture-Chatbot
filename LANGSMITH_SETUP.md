# LangSmith Observability Setup Guide

This guide explains how to set up and use LangSmith observability for the Agriculture Chatbot agentic flow.

## Overview

LangSmith provides end-to-end visibility into how your application handles requests. Each request generates a **trace**, which captures the full record of what happened. Within a trace are individual **runs**, the specific operations your application performed, such as an LLM call or a retrieval step.

## Prerequisites

1. **LangSmith Account**: Sign up or log in at [smith.langchain.com](https://smith.langchain.com)
2. **LangSmith API Key**: Follow the [Create an API key guide](https://docs.langchain.com/langsmith/observability-quickstart#prerequisites)
3. **Python Environment**: Make sure you have `langsmith` installed (already in `requirements.txt`)

## Setup Steps

### 1. Install Dependencies

The `langsmith` package is already included in `requirements.txt`. Install it if you haven't already:

```bash
pip install langsmith
```

### 2. Set Environment Variables

Add the following environment variables to your `.env` file:

```bash
# LangSmith Configuration
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_WORKSPACE_ID=your_workspace_id_here  # Optional: only needed if you have multiple workspaces
```

**How to get your API key:**
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Navigate to Settings → API Keys
3. Create a new API key or copy an existing one
4. Paste it into your `.env` file

**Workspace ID (Optional):**
- Only needed if your API key is linked to multiple workspaces
- Find it in the LangSmith UI under Settings → Workspace

### 3. Verify Setup

Run your application to verify LangSmith tracing is enabled:

```bash
python main.py
```

You should see output like:

```
📊 LangSmith Observability:
   ✅ Tracing: Enabled
   🔑 API Key: ✅ Set
   🏢 Workspace: Default
   📈 View traces at: https://smith.langchain.com
```

## What Gets Traced

The following components are instrumented with LangSmith tracing:

### 1. **LangGraph Nodes** (`main.py`)
- `classify_intent_node` - Intent classification step
- `rewrite_query_node` - Query rewriting for hybrid intents
- `retrieve_and_generate_node` - Document retrieval and response generation

### 2. **Intent Classifier** (`intent_classifier.py`)
- `intent_classifier.classify` - Classifies queries into disease/scheme/hybrid intents

### 3. **Query Rewriter** (`query_rewriter.py`)
- `query_rewriter.rewrite` - Rewrites hybrid queries into two focused queries

### 4. **Retrieval System** (`retrieve.py`)
- `retrieval_system.retrieve_disease` - Retrieves documents from disease knowledge base
- `retrieval_system.retrieve_scheme` - Retrieves documents from scheme knowledge base
- `retrieval_system.retrieve_and_generate` - Complete RAG pipeline

### 5. **Response Generator** (`response.py`)
- `response_generator.generate_response` - Generates farmer-friendly responses

## Viewing Traces

1. **Navigate to LangSmith UI**: Go to [smith.langchain.com](https://smith.langchain.com)
2. **Select Project**: Traces are sent to the "default" project (or your specified workspace)
3. **View Traces**: Each query execution creates a trace showing:
   - The complete workflow from intent classification to response generation
   - Individual LLM calls (Ollama model interactions)
   - Vector store retrieval operations
   - Timing information for each step
   - Inputs and outputs for each operation

## Trace Structure

Each trace contains:

```
┌─ classify_intent_node
│  └─ intent_classifier.classify
│     └─ [Ollama LLM Call]
│
├─ rewrite_query_node (if hybrid intent)
│  └─ query_rewriter.rewrite
│     └─ [Ollama LLM Call]
│
└─ retrieve_and_generate_node
   ├─ retrieval_system.retrieve_disease (if disease/hybrid)
   │  └─ [Vector Store Query]
   ├─ retrieval_system.retrieve_scheme (if scheme/hybrid)
   │  └─ [Vector Store Query]
   └─ response_generator.generate_response
      └─ [Ollama LLM Call]
```

## Benefits

1. **Debugging**: See exactly what happened at each step of your workflow
2. **Performance Monitoring**: Track latency for each operation
3. **Cost Tracking**: Monitor LLM usage and costs
4. **Quality Assurance**: Review inputs and outputs to improve prompts
5. **Error Tracking**: Identify where failures occur in the pipeline

## Troubleshooting

### Tracing Not Working

1. **Check Environment Variables**:
   ```bash
   echo $LANGSMITH_TRACING
   echo $LANGSMITH_API_KEY
   ```

2. **Verify Installation**:
   ```bash
   python -c "from langsmith import traceable; print('✅ langsmith installed')"
   ```

3. **Check Console Output**: The application will print whether tracing is enabled on startup

### No Traces Appearing in UI

1. **Wait a few seconds**: Traces may take a moment to appear
2. **Check API Key**: Ensure your API key is valid
3. **Check Workspace**: Make sure you're looking at the correct workspace/project
4. **Refresh**: Refresh the LangSmith UI

## Advanced Configuration

### Custom Project Name

You can send traces to a specific project by setting:

```bash
LANGSMITH_PROJECT=agriculture-chatbot
```

### Disable Tracing

To disable tracing (useful for local development):

```bash
LANGSMITH_TRACING=false
```

Or simply remove/comment out the environment variable.

## Additional Resources

- [LangSmith Documentation](https://docs.langchain.com/langsmith)
- [Tracing Quickstart](https://docs.langchain.com/langsmith/observability-quickstart)
- [LangGraph Tracing](https://docs.langchain.com/langgraph/how-tos/tracing)

## Support

For issues or questions:
- Check the [LangSmith Troubleshooting Guide](https://docs.langchain.com/langsmith/troubleshooting)
- Visit the [LangChain Discord](https://discord.gg/langchain)
- Review [LangSmith GitHub Issues](https://github.com/langchain-ai/langsmith/issues)

