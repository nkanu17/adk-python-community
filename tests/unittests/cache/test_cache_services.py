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

"""Unit tests for cache services."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from google.adk_community.cache.llm_response_cache import (
    LLMResponseCache,
    LLMResponseCacheConfig,
)
from google.adk_community.cache.providers.base import CacheEntry
from google.adk_community.cache.tool_cache import ToolCache, ToolCacheConfig


class TestLLMResponseCache:
  """Tests for LLMResponseCache."""

  @pytest.fixture
  def mock_provider(self):
    """Create a mock cache provider."""
    provider = MagicMock()
    provider.check = AsyncMock(return_value=None)
    provider.store = AsyncMock(return_value=None)
    return provider

  @pytest.fixture
  def mock_callback_context(self):
    """Create a mock callback context."""
    session = MagicMock()
    session.app_name = "test_app"
    session.user_id = "test_user"
    session.id = "test_session"
    session.events = []

    context = MagicMock()
    context.session = session
    return context

  @pytest.fixture
  def mock_llm_request(self):
    """Create a mock LLM request."""
    part = MagicMock()
    part.text = "What is Python?"

    content = MagicMock()
    content.role = "user"
    content.parts = [part]

    request = MagicMock()
    request.contents = [content]
    return request

  @pytest.mark.asyncio
  async def test_before_model_callback_cache_miss(
      self, mock_provider, mock_callback_context, mock_llm_request
  ):
    """Test before_model_callback with cache miss."""
    cache = LLMResponseCache(provider=mock_provider)
    result = await cache.before_model_callback(
        mock_callback_context, mock_llm_request
    )
    assert result is None
    mock_provider.check.assert_called_once()

  @pytest.mark.asyncio
  async def test_before_model_callback_cache_hit(
      self, mock_provider, mock_callback_context, mock_llm_request
  ):
    """Test before_model_callback with cache hit."""
    mock_provider.check.return_value = CacheEntry(
        prompt="What is Python?",
        response="Python is a programming language.",
    )
    cache = LLMResponseCache(provider=mock_provider)
    result = await cache.before_model_callback(
        mock_callback_context, mock_llm_request
    )
    assert result is not None
    assert result.content.parts[0].text == "Python is a programming language."

  @pytest.mark.asyncio
  async def test_before_model_callback_not_first_message(
      self, mock_provider, mock_callback_context, mock_llm_request
  ):
    """Test before_model_callback skips non-first messages."""
    # Add a user event to make it not the first message
    user_event = MagicMock()
    user_event.author = "user"
    mock_callback_context.session.events = [user_event]

    config = LLMResponseCacheConfig(first_message_only=True)
    cache = LLMResponseCache(provider=mock_provider, config=config)
    result = await cache.before_model_callback(
        mock_callback_context, mock_llm_request
    )
    assert result is None
    mock_provider.check.assert_not_called()

  @pytest.mark.asyncio
  async def test_after_model_callback_stores_response(
      self, mock_provider, mock_callback_context, mock_llm_request
  ):
    """Test after_model_callback stores response in cache."""
    cache = LLMResponseCache(provider=mock_provider)

    # First call before_model_callback to set up pending prompt
    await cache.before_model_callback(mock_callback_context, mock_llm_request)

    # Create mock response
    part = MagicMock()
    part.text = "Python is a programming language."
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    response = MagicMock()
    response.content = content
    response.error_message = None

    result = await cache.after_model_callback(mock_callback_context, response)
    assert result is None
    mock_provider.store.assert_called_once()


class TestToolCache:
  """Tests for ToolCache."""

  @pytest.fixture
  def mock_provider(self):
    """Create a mock cache provider."""
    provider = MagicMock()
    provider.check = AsyncMock(return_value=None)
    provider.store = AsyncMock(return_value=None)
    return provider

  @pytest.fixture
  def mock_tool_context(self):
    """Create a mock tool context."""
    invocation_context = MagicMock()
    invocation_context.app_name = "test_app"
    invocation_context.user_id = "test_user"
    invocation_context.session_id = "test_session"

    context = MagicMock()
    context.invocation_context = invocation_context
    return context

  @pytest.fixture
  def mock_tool(self):
    """Create a mock tool."""
    tool = MagicMock()
    tool.name = "test_tool"
    return tool

  @pytest.mark.asyncio
  async def test_before_tool_callback_cache_miss(
      self, mock_provider, mock_tool_context, mock_tool
  ):
    """Test before_tool_callback with cache miss."""
    config = ToolCacheConfig(tool_names={"test_tool"})
    cache = ToolCache(provider=mock_provider, config=config)
    args = {"query": "test query"}

    result = await cache.before_tool_callback(mock_tool, args, mock_tool_context)
    assert result is None
    mock_provider.check.assert_called_once()

  @pytest.mark.asyncio
  async def test_before_tool_callback_cache_hit(
      self, mock_provider, mock_tool_context, mock_tool
  ):
    """Test before_tool_callback with cache hit."""
    mock_provider.check.return_value = CacheEntry(
        prompt="test_tool:{\"query\": \"test query\"}",
        response='{"result": "cached result"}',
    )
    config = ToolCacheConfig(tool_names={"test_tool"})
    cache = ToolCache(provider=mock_provider, config=config)
    args = {"query": "test query"}

    result = await cache.before_tool_callback(mock_tool, args, mock_tool_context)
    assert result is not None
    assert result["result"] == "cached result"

  @pytest.mark.asyncio
  async def test_before_tool_callback_tool_not_in_list(
      self, mock_provider, mock_tool_context, mock_tool
  ):
    """Test before_tool_callback skips tools not in the list."""
    config = ToolCacheConfig(tool_names={"other_tool"})
    cache = ToolCache(provider=mock_provider, config=config)
    args = {"query": "test query"}

    result = await cache.before_tool_callback(mock_tool, args, mock_tool_context)
    assert result is None
    mock_provider.check.assert_not_called()

  @pytest.mark.asyncio
  async def test_after_tool_callback_stores_result(
      self, mock_provider, mock_tool_context, mock_tool
  ):
    """Test after_tool_callback stores result in cache."""
    config = ToolCacheConfig(tool_names={"test_tool"})
    cache = ToolCache(provider=mock_provider, config=config)
    args = {"query": "test query"}

    # First call before_tool_callback to set up pending call
    await cache.before_tool_callback(mock_tool, args, mock_tool_context)

    # Call after_tool_callback with result
    tool_response = {"result": "tool result"}
    result = await cache.after_tool_callback(
        mock_tool, args, mock_tool_context, tool_response
    )
    assert result is None
    mock_provider.store.assert_called_once()

  @pytest.mark.asyncio
  async def test_after_tool_callback_no_pending_call(
      self, mock_provider, mock_tool_context, mock_tool
  ):
    """Test after_tool_callback with no pending call."""
    config = ToolCacheConfig(tool_names={"test_tool"})
    cache = ToolCache(provider=mock_provider, config=config)
    args = {"query": "test query"}

    # Call after_tool_callback without before_tool_callback
    tool_response = {"result": "tool result"}
    result = await cache.after_tool_callback(
        mock_tool, args, mock_tool_context, tool_response
    )
    assert result is None
    mock_provider.store.assert_not_called()
