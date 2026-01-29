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

"""RedisVL semantic cache provider implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import BaseCacheProvider, CacheEntry

logger = logging.getLogger("google_adk." + __name__)


class RedisVLCacheProviderConfig(BaseModel):
  """Configuration for RedisVL cache provider.

  Attributes:
      redis_url: Redis connection string.
      name: Cache index name.
      ttl: Time-to-live in seconds for cached entries.
      distance_threshold: Semantic similarity threshold (0-2 for COSINE).
  """

  redis_url: str = Field(default="redis://localhost:6379")
  name: str = Field(default="adk_semantic_cache")
  ttl: int = Field(default=3600, ge=0)
  distance_threshold: float = Field(default=0.1, ge=0.0, le=2.0)


class RedisVLCacheProvider(BaseCacheProvider):
  """Cache provider using RedisVL's SemanticCache.

  This provider wraps RedisVL's synchronous SemanticCache with async
  wrappers using asyncio.to_thread().

  The vectorizer must be provided by the user and is not managed
  internally by this provider.
  """

  def __init__(
      self,
      config: RedisVLCacheProviderConfig,
      vectorizer: Any,
  ):
    """Initialize the RedisVL cache provider.

    Args:
        config: Configuration for the cache provider.
        vectorizer: A RedisVL vectorizer instance (e.g., OpenAITextVectorizer,
            HuggingFaceTextVectorizer). Must be provided by the user.
    """
    try:
      from redisvl.extensions.llmcache import SemanticCache
    except ImportError as e:
      raise ImportError(
          "redisvl is required for RedisVLCacheProvider. "
          "Install it with: pip install redisvl>=0.4.0"
      ) from e

    self._config = config
    self._vectorizer = vectorizer
    self._cache = SemanticCache(
        name=config.name,
        redis_url=config.redis_url,
        ttl=config.ttl,
        distance_threshold=config.distance_threshold,
        vectorizer=vectorizer,
    )

  async def check(
      self, prompt: str, **kwargs: Any
  ) -> Optional[CacheEntry]:
    """Check for a semantically similar prompt in the cache.

    Args:
        prompt: The prompt to check.
        **kwargs: Additional parameters (unused).

    Returns:
        A CacheEntry if a match is found, None otherwise.
    """
    result = await asyncio.to_thread(self._cache.check, prompt=prompt)
    if result:
      logger.debug("Cache hit for prompt: %s", prompt[:50])
      return CacheEntry(
          prompt=prompt,
          response=result[0]["response"],
          distance=result[0].get("vector_distance"),
      )
    logger.debug("Cache miss for prompt: %s", prompt[:50])
    return None

  async def store(
      self,
      prompt: str,
      response: str,
      metadata: Optional[Dict[str, Any]] = None,
      **kwargs: Any,
  ) -> None:
    """Store a prompt-response pair in the cache.

    Args:
        prompt: The prompt to cache.
        response: The response to cache.
        metadata: Optional metadata (currently unused by RedisVL).
        **kwargs: Additional parameters (unused).
    """
    await asyncio.to_thread(
        self._cache.store, prompt=prompt, response=response
    )
    logger.debug("Stored response for prompt: %s", prompt[:50])

  async def clear(self, **kwargs: Any) -> None:
    """Clear all entries from the cache."""
    await asyncio.to_thread(self._cache.clear)
    logger.info("Cache cleared")

  async def close(self) -> None:
    """Close the cache provider and release resources."""
    # RedisVL SemanticCache doesn't have an explicit close method
    # but we can disconnect the underlying Redis client if needed
    if hasattr(self._cache, "_index") and hasattr(
        self._cache._index, "client"
    ):
      await asyncio.to_thread(self._cache._index.client.close)
    logger.debug("RedisVL cache provider closed")

