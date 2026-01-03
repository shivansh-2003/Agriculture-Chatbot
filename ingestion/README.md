# Document Ingestion Pipeline

This folder contains the ingestion pipeline for Phase 1 of the Agriculture Chatbot project.

## Overview

The ingestion pipeline processes PDF documents and stores them in Pinecone vector database for efficient semantic search.

## Features

- **PDF Loading**: Uses LangChain's PyPDFLoader to parse PDF documents
- **Intelligent Chunking**: RecursiveCharacterTextSplitter with configurable chunk size (1000 tokens) and overlap (200 tokens)
- **Embeddings**: OpenAI text-embedding-3-small model for generating embeddings
- **Vector Storage**: Pinecone vector database with separate indexes for:
  - Citrus Pests & Diseases knowledge base
  - Government Schemes knowledge base
- **Metadata**: Each chunk includes source, collection type, page numbers, and chunk indices

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key for embeddings
- `PINECONE_API_KEY`: Your Pinecone API key

### 3. Get API Keys

**OpenAI API Key:**
- Sign up at https://platform.openai.com/
- Navigate to API Keys section
- Create a new API key

**Pinecone API Key:**
- Sign up at https://www.pinecone.io/
- Navigate to API Keys section
- Copy your API key

## Usage

Run the ingestion script:

```bash
python ingest.py
```

The script will:
1. Load both PDF documents
2. Chunk the documents with overlap
3. Generate embeddings
4. Create Pinecone indexes (if they don't exist)
5. Store embeddings with metadata

## Configuration

You can modify the following constants in `ingest.py`:

- `CHUNK_SIZE`: Size of each chunk (default: 1000 tokens)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200 tokens)
- `EMBEDDING_MODEL`: Embedding model to use (default: "text-embedding-3-small")
- `PINECONE_INDEX_NAME_CITRUS`: Name of Citrus diseases index
- `PINECONE_INDEX_NAME_SCHEMES`: Name of Government schemes index

## Output

After successful ingestion, you'll have two Pinecone indexes:

1. **citrus-diseases-pests**: Contains chunks from CitrusPlantPestsAndDiseases.pdf
2. **government-schemes**: Contains chunks from GovernmentSchemes.pdf

Each chunk includes metadata:
- `source`: PDF filename
- `collection`: Collection type (citrus_diseases_pests or government_schemes)
- `chunk_id`: Chunk index within the document
- `total_chunks`: Total number of chunks in the document
- `page_number`: Page number from the PDF (if available)

## Troubleshooting

**Error: PINECONE_API_KEY not found**
- Make sure you've created a `.env` file with your Pinecone API key

**Error: OPENAI_API_KEY not found**
- Make sure you've created a `.env` file with your OpenAI API key

**Error: PDF not found**
- Ensure `CitrusPlantPestsAndDiseases.pdf` and `GovernmentSchemes.pdf` are in the parent directory

**Index creation fails**
- Check your Pinecone account limits
- Ensure you have permission to create indexes
- Try using a different index name if there's a naming conflict

## Next Steps

After ingestion is complete, you can:
1. Use the Pinecone indexes for RAG queries
2. Implement intent detection and routing (Phase 2)
3. Build the FastAPI backend (Phase 3)

