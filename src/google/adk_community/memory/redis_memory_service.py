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

"""Redis-based memory service using RedisVL for vector search."""

from __future__ import annotations

import logging
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

from google.adk.memory import _utils
from google.adk.memory.base_memory_service import BaseMemoryService
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from .utils import extract_text_from_event

if TYPE_CHECKING:
  from google.adk.sessions.session import Session

logger = logging.getLogger("google_adk." + __name__)

try:
  from redisvl.index import SearchIndex
  from redisvl.query import VectorQuery
  from redisvl.utils.vectorize import BaseVectorizer

  REDISVL_AVAILABLE = True
except ImportError:
  REDISVL_AVAILABLE = False


class RedisMemoryService(BaseMemoryService):
  """Memory service implementation using RedisVL for vector search.

  This service uses Redis as a vector store for semantic memory search.
  Requires the 'redis-vector' optional dependency:

      pip install google-adk-community[redis-vector]

  See https://docs.redisvl.com/ for more information about RedisVL.
  """

  def __init__(
      self,
      vectorizer: BaseVectorizer,
      redis_url: str = "redis://localhost:6379",
      index_name: str = "adk_memories",
      config: Optional[RedisMemoryServiceConfig] = None,
  ):
    """Initializes the Redis memory service.

    Args:
        vectorizer: RedisVL vectorizer for embedding text (e.g., HFTextVectorizer).
        redis_url: Redis connection URL (default: redis://localhost:6379).
        index_name: Name for the Redis search index (default: adk_memories).
        config: RedisMemoryServiceConfig instance. If None, uses defaults.

    Raises:
        ImportError: If redisvl is not installed.
    """
    if not REDISVL_AVAILABLE:
      raise ImportError(
          "redisvl is required for RedisMemoryService. "
          "Install it with: pip install google-adk-community[redis-vector]"
      )

    self._vectorizer = vectorizer
    self._redis_url = redis_url
    self._index_name = index_name
    self._config = config or RedisMemoryServiceConfig()
    self._index: Optional[SearchIndex] = None
    self._initialized = False

  def _get_schema(self) -> dict[str, Any]:
    """Generate the RedisVL schema for memory storage."""
    return {
        "index": {
            "name": self._index_name,
            "prefix": f"{self._index_name}_docs",
        },
        "fields": [
            {"name": "content", "type": "text"},
            {"name": "app_name", "type": "tag"},
            {"name": "user_id", "type": "tag"},
            {"name": "session_id", "type": "tag"},
            {"name": "event_id", "type": "tag"},
            {"name": "author", "type": "tag"},
            {"name": "timestamp", "type": "numeric"},
            {
                "name": "content_embedding",
                "type": "vector",
                "attrs": {
                    "dims": self._vectorizer.dims,
                    "distance_metric": "cosine",
                    "algorithm": "hnsw",
                    "datatype": "float32",
                },
            },
        ],
    }

  async def _ensure_index(self):
    """Ensure the search index is created."""
    if self._initialized:
      return

    schema = self._get_schema()
    self._index = SearchIndex.from_dict(schema, redis_url=self._redis_url)

    try:
      await self._index.create(overwrite=False)
      logger.info("Created Redis search index: %s", self._index_name)
    except Exception as e:
      # Index might already exist
      logger.debug("Index creation skipped (may already exist): %s", e)

    self._initialized = True

  def _determine_salience(self, author: Optional[str]) -> float:
    """Determine salience score based on author."""
    if author == "user":
      return self._config.user_content_salience
    elif author == "model":
      return self._config.model_content_salience
    return self._config.default_salience

  async def _prepare_memory_data(
      self, event, content_text: str, session
  ) -> dict[str, Any]:
    """Prepare memory data structure for Redis storage."""
    timestamp_str = None
    if event.timestamp:
      timestamp_str = _utils.format_timestamp(event.timestamp)

    # Embed author and timestamp in content for better retrieval
    enriched_content = content_text
    metadata_parts = []
    if event.author:
      metadata_parts.append(f"Author: {event.author}")
    if timestamp_str:
      metadata_parts.append(f"Time: {timestamp_str}")

    if metadata_parts:
      metadata_prefix = "[" + ", ".join(metadata_parts) + "] "
      enriched_content = metadata_prefix + content_text

    # Generate embedding for the content
    embedding = await self._vectorizer.aembed(content_text)

    memory_data = {
        "content": enriched_content,
        "app_name": session.app_name,
        "user_id": session.user_id,
        "session_id": session.id,
        "event_id": event.id,
        "author": event.author or "",
        "timestamp": event.timestamp or 0,
        "content_embedding": embedding,
    }

    return memory_data

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to Redis vector store."""
    await self._ensure_index()
    memories_added = 0

    for event in session.events:
      content_text = extract_text_from_event(event)
      if not content_text:
        continue

      try:
        memory_data = await self._prepare_memory_data(
            event, content_text, session
        )

        # Load data into Redis
        assert self._index is not None
        keys = self._index.load([memory_data])
        memories_added += 1
        logger.debug("Added memory for event %s with key %s", event.id, keys[0])
      except Exception as e:
        logger.error(
            "Failed to add memory for event %s due to error: %s", event.id, e
        )

    logger.info("Added %d memories from session %s", memories_added, session.id)

  def _convert_to_memory_entry(self, result: dict) -> Optional[MemoryEntry]:
    """Convert Redis search result to MemoryEntry."""
    try:
      content_text = result.get("content", "")

      # Extract author and timestamp from enriched content format
      author = result.get("author", "")
      timestamp = result.get("timestamp", 0)

      # Remove metadata prefix from content if present
      clean_content = content_text
      if content_text.startswith("["):
        bracket_end = content_text.find("]")
        if bracket_end != -1:
          clean_content = content_text[bracket_end + 2 :]

      content = types.Content(parts=[types.Part(text=clean_content)])

      return MemoryEntry(
          content=content,
          author=author if author else None,
          timestamp=timestamp if timestamp else None,
      )
    except (KeyError, ValueError) as e:
      logger.debug("Failed to convert result to MemoryEntry: %s", e)
      return None

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories using Redis vector search."""
    try:
      await self._ensure_index()

      # Generate query embedding
      query_embedding = await self._vectorizer.aembed(query)

      # Build filter expression for app_name and user_id
      filter_expr = f"@app_name:{{{app_name}}} @user_id:{{{user_id}}}"

      # Create vector query
      vector_query = VectorQuery(
          vector=query_embedding,
          vector_field_name="content_embedding",
          return_fields=["content", "author", "timestamp", "vector_distance"],
          num_results=self._config.search_top_k,
          filter_expression=filter_expr,
      )

      # Execute search
      assert self._index is not None
      results = self._index.query(vector_query)

      memories = []
      for result in results:
        memory_entry = self._convert_to_memory_entry(result)
        if memory_entry:
          memories.append(memory_entry)

      logger.info("Found %d memories for query: '%s'", len(memories), query)
      return SearchMemoryResponse(memories=memories)

    except Exception as e:
      logger.error("Failed to search memories due to error: %s", e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service and cleanup resources."""
    if self._index:
      # RedisVL handles connection cleanup internally
      pass


class RedisMemoryServiceConfig(BaseModel):
  """Configuration for Redis memory service behavior.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per search.
      user_content_salience: Salience for user-authored content (0.0-1.0).
      model_content_salience: Salience for model-generated content (0.0-1.0).
      default_salience: Default salience value for memories (0.0-1.0).
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  user_content_salience: float = Field(default=0.8, ge=0.0, le=1.0)
  model_content_salience: float = Field(default=0.7, ge=0.0, le=1.0)
  default_salience: float = Field(default=0.6, ge=0.0, le=1.0)
