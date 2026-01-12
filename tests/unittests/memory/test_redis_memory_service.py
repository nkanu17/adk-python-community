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

"""Tests for RedisMemoryService."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from unittest.mock import PropertyMock

from google.adk.events.event import Event
from google.adk.sessions.session import Session
from google.genai import types
import pytest

# Skip all tests if redisvl is not installed
pytest.importorskip('redisvl')

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'
MOCK_SESSION_ID = 'session-1'

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id='event-1',
            invocation_id='inv-1',
            author='user',
            timestamp=12345,
            content=types.Content(
                parts=[types.Part(text='Hello, I like Python.')]
            ),
        ),
        Event(
            id='event-2',
            invocation_id='inv-2',
            author='model',
            timestamp=12346,
            content=types.Content(
                parts=[
                    types.Part(text='Python is a great programming language.')
                ]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id='event-3',
            invocation_id='inv-3',
            author='user',
            timestamp=12347,
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
)


@pytest.fixture
def mock_vectorizer():
  """Mock RedisVL vectorizer."""
  vectorizer = MagicMock()
  vectorizer.dims = 384  # Common embedding dimension
  vectorizer.aembed = AsyncMock(return_value=[0.1] * 384)
  return vectorizer


@pytest.fixture
def mock_search_index():
  """Mock RedisVL SearchIndex."""
  # Mock at the redisvl.index level instead
  with patch('redisvl.index.SearchIndex') as mock_index_class:
    mock_index = MagicMock()
    mock_index.create = AsyncMock()
    mock_index.load = MagicMock(return_value=['key1'])
    mock_index.query = MagicMock(return_value=[])
    mock_index_class.from_dict = MagicMock(return_value=mock_index)
    yield mock_index


@pytest.fixture
def memory_service(mock_vectorizer, mock_search_index):
  """Create RedisMemoryService instance for testing."""
  try:
    from google.adk_community.memory.redis_memory_service import RedisMemoryService

    return RedisMemoryService(
        vectorizer=mock_vectorizer,
        redis_url='redis://localhost:6379',
        index_name='test_index',
    )
  except ImportError:
    pytest.skip('redisvl not installed')


@pytest.fixture
def memory_service_with_config(mock_vectorizer, mock_search_index):
  """Create RedisMemoryService with custom config."""
  try:
    from google.adk_community.memory.redis_memory_service import RedisMemoryService
    from google.adk_community.memory.redis_memory_service import RedisMemoryServiceConfig

    config = RedisMemoryServiceConfig(
        search_top_k=5, user_content_salience=0.9, model_content_salience=0.6
    )
    return RedisMemoryService(
        vectorizer=mock_vectorizer,
        redis_url='redis://localhost:6379',
        index_name='test_index',
        config=config,
    )
  except ImportError:
    pytest.skip('redisvl not installed')


class TestRedisMemoryServiceConfig:
  """Tests for RedisMemoryServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    try:
      from google.adk_community.memory.redis_memory_service import RedisMemoryServiceConfig

      config = RedisMemoryServiceConfig()
      assert config.search_top_k == 10
      assert config.user_content_salience == 0.8
      assert config.model_content_salience == 0.7
      assert config.default_salience == 0.6
    except ImportError:
      pytest.skip('redisvl not installed')

  def test_custom_config(self):
    """Test custom configuration values."""
    try:
      from google.adk_community.memory.redis_memory_service import RedisMemoryServiceConfig

      config = RedisMemoryServiceConfig(
          search_top_k=20,
          user_content_salience=0.9,
          model_content_salience=0.75,
          default_salience=0.5,
      )
      assert config.search_top_k == 20
      assert config.user_content_salience == 0.9
      assert config.model_content_salience == 0.75
      assert config.default_salience == 0.5
    except ImportError:
      pytest.skip('redisvl not installed')


class TestRedisMemoryService:
  """Tests for RedisMemoryService."""

  @pytest.mark.asyncio
  async def test_add_session_to_memory_success(
      self, memory_service, mock_search_index, mock_vectorizer
  ):
    """Test successful addition of session memories."""
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should call load twice (one per valid event)
    assert mock_search_index.load.call_count == 2

    # Verify vectorizer was called for each event
    assert mock_vectorizer.aembed.call_count == 2

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_search_index
  ):
    """Test that events without content are filtered out."""
    await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

    # Should make 0 load calls (no valid events)
    assert mock_search_index.load.call_count == 0

  @pytest.mark.asyncio
  async def test_search_memory_success(
      self, memory_service, mock_search_index, mock_vectorizer
  ):
    """Test successful memory search."""
    # Mock search results
    mock_search_index.query.return_value = [
        {
            'content': (
                '[Author: user, Time: 2025-01-01T00:00:00] Python is great'
            ),
            'author': 'user',
            'timestamp': 12345,
        },
        {
            'content': (
                '[Author: model, Time: 2025-01-01T00:01:00] I like programming'
            ),
            'author': 'model',
            'timestamp': 12346,
        },
    ]

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='Python programming'
    )

    # Should return 2 memories
    assert len(result.memories) == 2
    assert result.memories[0].author == 'user'
    assert result.memories[1].author == 'model'

    # Verify vectorizer was called to embed the query
    mock_vectorizer.aembed.assert_called_once_with('Python programming')

  @pytest.mark.asyncio
  async def test_search_memory_respects_top_k(
      self, memory_service_with_config, mock_search_index, mock_vectorizer
  ):
    """Test that config.search_top_k is used."""
    mock_search_index.query.return_value = []

    await memory_service_with_config.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='test query'
    )

    # Verify query was called with correct num_results
    call_args = mock_search_index.query.call_args
    query_obj = call_args[0][0]
    assert query_obj.num_results == 5  # Custom config value

  @pytest.mark.asyncio
  async def test_search_memory_error_handling(
      self, memory_service, mock_search_index
  ):
    """Test graceful error handling during memory search."""
    mock_search_index.query.side_effect = Exception('Redis Error')

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME, user_id=MOCK_USER_ID, query='test query'
    )

    # Should return empty results on error
    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_add_session_error_handling(
      self, memory_service, mock_search_index
  ):
    """Test error handling during memory addition."""
    mock_search_index.load.side_effect = Exception('Redis Error')

    # Should not raise exception, just log error
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should still attempt to load for each valid event
    assert mock_search_index.load.call_count == 2

  @pytest.mark.asyncio
  async def test_close(self, memory_service):
    """Test close method."""
    # Should not raise exception
    await memory_service.close()

  def test_import_error_without_redisvl(self):
    """Test that ImportError is raised when redisvl is not installed."""
    with patch(
        'google.adk_community.memory.redis_memory_service.REDISVL_AVAILABLE',
        False,
    ):
      try:
        from google.adk_community.memory.redis_memory_service import RedisMemoryService

        mock_vectorizer = MagicMock()
        with pytest.raises(ImportError, match='redisvl is required'):
          RedisMemoryService(vectorizer=mock_vectorizer)
      except ImportError:
        pytest.skip('redisvl not installed')
