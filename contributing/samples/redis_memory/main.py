# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of using RedisMemoryService with get_fast_api_app."""

import os
from urllib.parse import urlparse

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.cli.service_registry import get_service_registry
from redisvl.utils.vectorize import HFTextVectorizer

from google.adk_community.memory.redis_memory_service import (
    RedisMemoryService,
    RedisMemoryServiceConfig,
)

# Load environment variables from .env file if it exists
load_dotenv()


def redis_memory_factory(uri: str, **kwargs):
  """Factory function for creating RedisMemoryService from URI.

  URI format: redis-memory://localhost:6379/index_name
  """
  parsed = urlparse(uri)
  host = parsed.hostname or "localhost"
  port = parsed.port or 6379
  index_name = parsed.path.lstrip("/") or "adk_memories"

  redis_url = f"redis://{host}:{port}"

  # Get vectorizer model from environment or use default
  model_name = os.getenv(
      "REDIS_MEMORY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
  )

  # Create vectorizer
  vectorizer = HFTextVectorizer(model=model_name)

  # Create config from environment
  config = RedisMemoryServiceConfig(
      search_top_k=int(os.getenv("REDIS_MEMORY_TOP_K", "10")),
  )

  return RedisMemoryService(
      vectorizer=vectorizer,
      redis_url=redis_url,
      index_name=index_name,
      config=config,
  )


# Register the Redis memory service factory
get_service_registry().register_memory_service("redis-memory", redis_memory_factory)

# Build Redis Memory URI from environment variables
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_index = os.getenv("REDIS_MEMORY_INDEX", "adk_memories")
MEMORY_SERVICE_URI = f"redis-memory://{redis_host}:{redis_port}/{redis_index}"

# Create the FastAPI app using get_fast_api_app
app: FastAPI = get_fast_api_app(
    agents_dir=".",
    memory_service_uri=MEMORY_SERVICE_URI,
    web=True,
)


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 8000))
  print(f"""
Redis Memory Sample Starting...
Memory Service URI: {MEMORY_SERVICE_URI}
Web UI: http://localhost:{port}
""")
  uvicorn.run(app, host="0.0.0.0", port=port)

