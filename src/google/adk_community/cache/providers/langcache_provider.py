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

"""LangCache semantic cache provider implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import BaseCacheProvider, CacheEntry

logger = logging.getLogger("google_adk." + __name__)


class LangCacheProviderConfig(BaseModel):
  """Configuration for LangCache provider.

  Attributes:
      name: Cache name identifier.
      ttl: Time-to-live in seconds for cached entries.
      similarity_threshold: Semantic similarity threshold (0-1).
  """

  name: str = Field(default="adk_langcache")
  ttl: int = Field(default=3600, ge=0)
  similarity_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class LangCacheProvider(BaseCacheProvider):
  """Cache provider using LangCache managed service.

  LangCache is a managed semantic caching service with native async
  support. It uses Redis Cloud infrastructure.
  """

  def __init__(self, config: LangCacheProviderConfig):
    """Initialize the LangCache provider.

    Args:
        config: Configuration for the cache provider.
    """
    try:
      from langcache import LangCache
    except ImportError as e:
      raise ImportError(
          "langcache is required for LangCacheProvider. "
          "Install it with: pip install langcache>=0.8.0"
      ) from e

    self._config = config
    self._cache = LangCache(name=config.name, ttl=config.ttl)

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
    result = await self._cache.search_async(
        prompt, threshold=self._config.similarity_threshold
    )
    if result:
      logger.debug("Cache hit for prompt: %s", prompt[:50])
      return CacheEntry(
          prompt=prompt,
          response=result.response,
          distance=getattr(result, "distance", None),
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
        metadata: Optional metadata (currently unused by LangCache).
        **kwargs: Additional parameters (unused).
    """
    await self._cache.set_async(prompt=prompt, response=response)
    logger.debug("Stored response for prompt: %s", prompt[:50])

  async def clear(self, **kwargs: Any) -> None:
    """Clear all entries from the cache."""
    if hasattr(self._cache, "clear_async"):
      await self._cache.clear_async()
    elif hasattr(self._cache, "clear"):
      self._cache.clear()
    logger.info("Cache cleared")

  async def close(self) -> None:
    """Close the cache provider and release resources."""
    if hasattr(self._cache, "close_async"):
      await self._cache.close_async()
    elif hasattr(self._cache, "close"):
      self._cache.close()
    logger.debug("LangCache provider closed")

