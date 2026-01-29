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

"""Unit tests for cache providers."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from google.adk_community.cache.providers.base import BaseCacheProvider, CacheEntry


class TestCacheEntry:
  """Tests for CacheEntry model."""

  def test_cache_entry_creation(self):
    """Test creating a CacheEntry with all fields."""
    entry = CacheEntry(
        prompt="What is Python?",
        response="Python is a programming language.",
        metadata={"source": "test"},
        distance=0.05,
    )
    assert entry.prompt == "What is Python?"
    assert entry.response == "Python is a programming language."
    assert entry.metadata == {"source": "test"}
    assert entry.distance == 0.05

  def test_cache_entry_optional_fields(self):
    """Test CacheEntry with optional fields."""
    entry = CacheEntry(
        prompt="Hello",
        response="Hi there!",
    )
    assert entry.prompt == "Hello"
    assert entry.response == "Hi there!"
    assert entry.metadata is None
    assert entry.distance is None


class TestRedisVLCacheProvider:
  """Tests for RedisVLCacheProvider."""

  @pytest.fixture
  def mock_semantic_cache(self):
    """Create a mock SemanticCache."""
    mock = MagicMock()
    mock.check.return_value = []
    mock.store.return_value = None
    mock.clear.return_value = None
    return mock

  @pytest.fixture
  def mock_vectorizer(self):
    """Create a mock vectorizer."""
    return MagicMock()

  @pytest.fixture
  def mock_redisvl_module(self, mock_semantic_cache):
    """Create a mock redisvl module."""
    mock_module = MagicMock()
    mock_module.SemanticCache = MagicMock(return_value=mock_semantic_cache)
    return mock_module

  @pytest.mark.asyncio
  async def test_check_cache_miss(
      self, mock_semantic_cache, mock_vectorizer, mock_redisvl_module
  ):
    """Test cache check with no match."""
    mock_semantic_cache.check.return_value = []

    with patch.dict(
        sys.modules, {"redisvl.extensions.llmcache": mock_redisvl_module}
    ):
      # Force reimport to pick up the mock
      import importlib
      import google.adk_community.cache.providers.redisvl_provider as redisvl_mod

      importlib.reload(redisvl_mod)

      config = redisvl_mod.RedisVLCacheProviderConfig(
          redis_url="redis://localhost:6379"
      )
      provider = redisvl_mod.RedisVLCacheProvider(
          config=config, vectorizer=mock_vectorizer
      )

      result = await provider.check("What is Python?")
      assert result is None

  @pytest.mark.asyncio
  async def test_check_cache_hit(
      self, mock_semantic_cache, mock_vectorizer, mock_redisvl_module
  ):
    """Test cache check with a match."""
    mock_semantic_cache.check.return_value = [
        {
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
            "vector_distance": 0.05,
        }
    ]

    with patch.dict(
        sys.modules, {"redisvl.extensions.llmcache": mock_redisvl_module}
    ):
      import importlib
      import google.adk_community.cache.providers.redisvl_provider as redisvl_mod

      importlib.reload(redisvl_mod)

      config = redisvl_mod.RedisVLCacheProviderConfig(
          redis_url="redis://localhost:6379"
      )
      provider = redisvl_mod.RedisVLCacheProvider(
          config=config, vectorizer=mock_vectorizer
      )

      result = await provider.check("What is Python?")
      assert result is not None
      assert result.response == "Python is a programming language."
      assert result.distance == 0.05

  @pytest.mark.asyncio
  async def test_store(
      self, mock_semantic_cache, mock_vectorizer, mock_redisvl_module
  ):
    """Test storing a prompt-response pair."""
    with patch.dict(
        sys.modules, {"redisvl.extensions.llmcache": mock_redisvl_module}
    ):
      import importlib
      import google.adk_community.cache.providers.redisvl_provider as redisvl_mod

      importlib.reload(redisvl_mod)

      config = redisvl_mod.RedisVLCacheProviderConfig(
          redis_url="redis://localhost:6379"
      )
      provider = redisvl_mod.RedisVLCacheProvider(
          config=config, vectorizer=mock_vectorizer
      )

      await provider.store("What is Python?", "Python is a language.")
      mock_semantic_cache.store.assert_called_once()

  @pytest.mark.asyncio
  async def test_clear(
      self, mock_semantic_cache, mock_vectorizer, mock_redisvl_module
  ):
    """Test clearing the cache."""
    with patch.dict(
        sys.modules, {"redisvl.extensions.llmcache": mock_redisvl_module}
    ):
      import importlib
      import google.adk_community.cache.providers.redisvl_provider as redisvl_mod

      importlib.reload(redisvl_mod)

      config = redisvl_mod.RedisVLCacheProviderConfig(
          redis_url="redis://localhost:6379"
      )
      provider = redisvl_mod.RedisVLCacheProvider(
          config=config, vectorizer=mock_vectorizer
      )

      await provider.clear()
      mock_semantic_cache.clear.assert_called_once()


class TestLangCacheProvider:
  """Tests for LangCacheProvider."""

  @pytest.fixture
  def mock_langcache_instance(self):
    """Create a mock LangCache instance."""
    mock = MagicMock()
    mock.search_async = AsyncMock(return_value=None)
    mock.set_async = AsyncMock(return_value=None)
    mock.clear_async = AsyncMock(return_value=None)
    return mock

  @pytest.fixture
  def mock_langcache_module(self, mock_langcache_instance):
    """Create a mock langcache module."""
    mock_module = MagicMock()
    mock_module.LangCache = MagicMock(return_value=mock_langcache_instance)
    return mock_module

  @pytest.mark.asyncio
  async def test_check_cache_miss(
      self, mock_langcache_instance, mock_langcache_module
  ):
    """Test cache check with no match."""
    mock_langcache_instance.search_async.return_value = None

    with patch.dict(sys.modules, {"langcache": mock_langcache_module}):
      import importlib
      import google.adk_community.cache.providers.langcache_provider as lc_mod

      importlib.reload(lc_mod)

      config = lc_mod.LangCacheProviderConfig()
      provider = lc_mod.LangCacheProvider(config=config)

      result = await provider.check("What is Python?")
      assert result is None

  @pytest.mark.asyncio
  async def test_check_cache_hit(
      self, mock_langcache_instance, mock_langcache_module
  ):
    """Test cache check with a match."""
    mock_langcache_instance.search_async.return_value = MagicMock(
        response="Python is a programming language.",
        similarity=0.95,
    )

    with patch.dict(sys.modules, {"langcache": mock_langcache_module}):
      import importlib
      import google.adk_community.cache.providers.langcache_provider as lc_mod

      importlib.reload(lc_mod)

      config = lc_mod.LangCacheProviderConfig()
      provider = lc_mod.LangCacheProvider(config=config)

      result = await provider.check("What is Python?")
      assert result is not None
      assert result.response == "Python is a programming language."

  @pytest.mark.asyncio
  async def test_store(self, mock_langcache_instance, mock_langcache_module):
    """Test storing a prompt-response pair."""
    with patch.dict(sys.modules, {"langcache": mock_langcache_module}):
      import importlib
      import google.adk_community.cache.providers.langcache_provider as lc_mod

      importlib.reload(lc_mod)

      config = lc_mod.LangCacheProviderConfig()
      provider = lc_mod.LangCacheProvider(config=config)

      await provider.store("What is Python?", "Python is a language.")
      mock_langcache_instance.set_async.assert_called_once()

