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

"""Agent definition for Redis Memory sample."""

from datetime import datetime

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_memory
from google.adk.tools import preload_memory


def update_current_time(callback_context: CallbackContext):
  """Update the current time in the agent's state."""
  callback_context.state["_time"] = datetime.now().isoformat()


root_agent = Agent(
    model="gemini-2.5-flash",
    name="redis_memory_agent",
    description="Agent with Redis-backed vector memory for semantic search.",
    before_agent_callback=update_current_time,
    instruction=(
        "You are a helpful assistant with access to memory tools.\n"
        "You can remember information from past conversations using Redis vector search.\n\n"
        "When the user asks about something you discussed before, use the load_memory tool "
        "to search for relevant information from past conversations.\n"
        "If the first search does not find relevant information, try different search terms.\n\n"
        "Current time: {_time}"
    ),
    tools=[preload_memory, load_memory],
)

