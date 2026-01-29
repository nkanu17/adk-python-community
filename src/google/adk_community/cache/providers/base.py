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

"""Base cache provider interface for semantic caching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel


class CacheEntry(BaseModel):
  """Represents a cached entry with prompt, response, and metadata.

  Attributes:
      prompt: The original prompt that was cached.
      response: The cached response for the prompt.
      metadata: Optional metadata associated with the cache entry.
      distance: Optional semantic distance score from the query.
  """

  prompt: str
  response: str
  metadata: Optional[Dict[str, Any]] = None
  distance: Optional[float] = None


class BaseCacheProvider(ABC):
  """Abstract base class for semantic cache providers.

  This interface defines the contract for cache providers that support
  semantic similarity matching for prompts and responses.
  """

  @abstractmethod
  async def check(
      self, prompt: str, **kwargs: Any
  ) -> Optional[CacheEntry]:
    """Check if a semantically similar prompt exists in the cache.

    Args:
        prompt: The prompt to check for semantic matches.
        **kwargs: Additional provider-specific parameters.

    Returns:
        A CacheEntry if a semantic match is found, None otherwise.
    """
    ...

  @abstractmethod
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
        metadata: Optional metadata to store with the entry.
        **kwargs: Additional provider-specific parameters.
    """
    ...

  @abstractmethod
  async def clear(self, **kwargs: Any) -> None:
    """Clear all entries from the cache.

    Args:
        **kwargs: Additional provider-specific parameters.
    """
    ...

  @abstractmethod
  async def close(self) -> None:
    """Close the cache provider and release resources."""
    ...
