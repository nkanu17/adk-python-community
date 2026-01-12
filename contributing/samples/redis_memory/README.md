# Redis Memory Sample

This sample demonstrates how to use Redis with RedisVL as a vector memory backend
for ADK agents using the community package.

## Prerequisites

- Python 3.9+ (Python 3.11+ recommended)
- Redis Stack (with vector search support) or Redis 8.0+
- ADK and ADK Community installed

## Setup

### 1. Install Dependencies

```bash
pip install google-adk google-adk-community[redis-vl]
```

This installs RedisVL and sentence-transformers for vector embeddings.

### 2. Set Up Redis Stack

**Option A: Docker (Recommended)**

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

**Option B: Redis Cloud**

Create a free Redis Cloud account at [redis.io/cloud](https://redis.io/cloud) and use the connection URL.

### 3. Configure Environment Variables

Create a `.env` file in this directory:

```bash
# Required: Google API key for the agent
GOOGLE_API_KEY=your-google-api-key

# Optional: Redis connection (defaults to localhost:6379)
REDIS_HOST=localhost
REDIS_PORT=6379

# Optional: Memory index name (defaults to adk_memories)
REDIS_MEMORY_INDEX=adk_memories

# Optional: Embedding model (defaults to sentence-transformers/all-MiniLM-L6-v2)
REDIS_MEMORY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional: Number of memories to retrieve (defaults to 10)
REDIS_MEMORY_TOP_K=10
```

## Usage

### Option 1: Using `main.py` with FastAPI (Recommended)

```bash
python main.py
```

This starts a FastAPI server with the ADK web UI at `http://localhost:8000`.

### Option 2: Using `Runner` Directly

```python
from redisvl.utils.vectorize import HFTextVectorizer
from google.adk.runners import Runner
from google.adk_community.memory.redis_memory_service import (
    RedisMemoryService,
    RedisMemoryServiceConfig,
)

# Create vectorizer
vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")

# Create memory service
memory_service = RedisMemoryService(
    vectorizer=vectorizer,
    redis_url="redis://localhost:6379",
    index_name="adk_memories",
)

# Use with runner
runner = Runner(
    app_name="my_app",
    agent=root_agent,
    memory_service=memory_service,
)
```

### Advanced Configuration

```python
from google.adk_community.memory.redis_memory_service import (
    RedisMemoryService,
    RedisMemoryServiceConfig,
)

config = RedisMemoryServiceConfig(
    search_top_k=20,              # Retrieve more memories per query
    user_content_salience=0.9,    # Higher importance for user messages
    model_content_salience=0.75,  # Medium importance for model responses
)

memory_service = RedisMemoryService(
    vectorizer=vectorizer,
    redis_url="redis://localhost:6379",
    index_name="adk_memories",
    config=config,
)
```

## Sample Structure

```
redis_memory/
├── main.py                    # FastAPI server using get_fast_api_app
├── redis_memory_agent/
│   ├── __init__.py            # Agent package initialization
│   └── agent.py               # Agent definition with memory tools
└── README.md                  # This file
```

## Sample Queries

Try these queries to test the memory functionality:

**Session 1:**
- "Hello, my name is Alex and I'm a software engineer"
- "I work on machine learning projects using Python and TensorFlow"
- "My favorite programming language is Rust for systems programming"

**Session 2 (new session):**
- "What do you know about me?"
- "What programming languages do I use?"

The agent should recall information from the previous session using Redis vector search.

## Configuration Options

### RedisMemoryServiceConfig

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `search_top_k` | int | 10 | Maximum memories to retrieve per search |
| `user_content_salience` | float | 0.8 | Importance weight for user messages |
| `model_content_salience` | float | 0.7 | Importance weight for model responses |
| `default_salience` | float | 0.6 | Fallback importance value |

## Features

Redis Memory Service provides:

- **Vector similarity search**: Semantic search using embeddings
- **Automatic indexing**: Creates and manages Redis vector index
- **Session-aware**: Stores memories with session and user context
- **Configurable embeddings**: Use any HuggingFace embedding model
- **High performance**: Redis in-memory speed for fast retrieval

## Learn More

- [RedisVL Documentation](https://redisvl.com/)
- [Redis Vector Search](https://redis.io/docs/stack/search/reference/vectors/)
- [ADK Memory Documentation](https://google.github.io/adk-docs)

