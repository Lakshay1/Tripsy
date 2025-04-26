"""
Skeleton orchestrator agent that demonstrates Anthropic tool‑use (function calling).

▸ Requires the `anthropic` Python SDK ≥ 0.23.0
   pip3 install anthropic
   pip3 install google-api-python-client google-auth-oauthlib google-auth-httplib2

▸ Make sure the `ANTHROPIC_API_KEY` environment variable is set before running:
   export ANTHROPIC_API_KEY="sk‑..."

The orchestrator keeps the chat loop alive until Claude no longer wants to call a
client‑side tool. It extracts tool‑use blocks, executes the mapped Python
function, feeds the result back to Claude with a `tool_result` block, and
finally prints the model’s answer.

Replace the dummy `get_weather` implementation with a real API call (or register
additional tools) to build richer agents.
"""
from __future__ import annotations

import os
import pdb
import sys
from typing import Any, Callable, Dict, List

import anthropic # type: ignore

from tools.email_tool.tool import fetch_emails

BASE_PROMPT = '/Users/lakshayk/Developer/Tripsy/Tripsy/agent/base_prompt.txt'

# ---------------------------------------------------------------------------
# Core orchestrator class
# ---------------------------------------------------------------------------

class AnthropicOrchestrator:
    """Minimal orchestrator that handles Anthropic tool‑use messages."""

    def __init__(
        self,
        *,
        model: str = "claude-3-7-sonnet-20250219",  # ⇠ pick your favourite C3 tier
        tools: List[dict] | None = None,
        api_key: str | None = None,
        max_tokens: int = 64000,
    ) -> None:
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.tools: List[dict] = tools or []
        self._registry: Dict[str, Callable[..., Any]] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def register_tool(self, tool_def: dict, python_fn: Callable[..., Any]) -> None:
        """Attach a JSON tool definition **and** the Python function that executes it."""
        name = tool_def["name"]
        assert callable(python_fn), "python_fn must be a callable"
        self.tools.append(tool_def)
        self._registry[name] = python_fn

    def chat(self, user_prompt: str) -> anthropic.types.Message:
        """High‑level helper: send `user_prompt`, run tools if requested, return final msg."""
        messages: List[dict] = [{"role": "user", "content": user_prompt}]

        while True:
            response = self._send(messages)
            # pdb.set_trace()
            if response.stop_reason == "tool_use":
                # Claude wants one or more tools. Iterate over tool_use blocks.
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    tool_use_id = block.id
                    tool_name = block.name
                    tool_args = block.input

                    result, error_flag = self._execute_local_tool(tool_name, tool_args)

                    # Tell Claude the result so it can compose its answer.
                    messages.extend(
                        [
                            {"role": "assistant", "content": response.content},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use_id,
                                        "content": str(result),
                                        **({"is_error": True} if error_flag else {}),
                                    }
                                ],
                            },
                        ]
                    )
                # Loop continues – Claude may want another tool call or produce final answer.
                continue

            # Any other stop_reason means we’re done ("stop_sequence", "max_tokens", etc.)
            return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, messages: List[dict]) -> anthropic.types.Message:
        """Wrapper around anthropic.messages.create()."""
        return self.client.messages.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            # Uncomment for token‑efficient tools beta (if you have access):
            # headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
        )

    def _execute_local_tool(self, name: str, kwargs: dict) -> tuple[Any, bool]:
        """Look up and run the registered python function. Returns (result, is_error)."""
        fn = self._registry.get(name)
        if fn is None:
            return f"No local implementation found for tool '{name}'", True
        try:
            return fn(**kwargs), False
        except Exception as exc:  # ⬅️ never let an exception kill the orchestrator
            return f"Tool execution failed: {exc}", True


# ---------------------------------------------------------------------------
# Demo: a trivial weather tool
# ---------------------------------------------------------------------------

# def fetch_emails(location: str, unit: str = "celsius") -> str:
#     """Dummy weather implementation (replace with a real API like OpenWeather)."""
#     return f"The current weather in {location} is 15° {unit}. (stub)"

EMAIL_TOOL_DEF = {
    "name": "fetch_emails",
    "description": (
        "Gets emails from the user's gmail inbox using filters to find relevant bookings "
        "for flights, hotels, and other travel-related items."
        "Please use following parameters to filter the emails: " \
        "1. title_keywords=['itinerary', 'booking', 'reservation', 'airline', 'emirates', 'united', 'air india'], " \
        "2. content_keywords=['trip', 'fly', 'booking', 'hotel', 'itinerary', 'flight', 'reservation', 'airbnb', 'booking']" \
        "3. start_date: Always set it to 2025/03/05" \
        "4. end_date: Always set it to 2025/03/15" \
        "Example invocation: - fetch_emails(labels=['INBOX', 'UPDATES'], start_date='2025/03/09', end_date='2025/03/20', title_keywords=['itinerary', 'booking'], " \
                         "content_keywords=['emirates', 'trip', 'fly', 'booking', 'hotel', 'itinerary', 'flight', 'reservation', 'airbnb', 'booking'])"
        ),
    "input_schema": {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of labels to filter emails by (e.g. ['travel', 'bookings', 'flights', 'location']).",
            },
            "start_date": {
                "type": "string",
                "description": "Filter to specify start date from where emails should be read from.",
            },
            "end_date": {
                "type": "string",
                "description": "Filter to specify end date until where emails should be read from.",
            },
            "title_keywords": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of keywords to filter emails by (e.g. ['flight', 'hotel', 'booking']).",
            },
            "content_keywords": {
                "type": "array",
                "items": { "type": "string" },
                "description": "List of keywords to filter emails by (e.g. ['flight', 'hotel', 'booking']). Defaults to []",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of emails to return. Defaults to 50.",
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Enter a string, or type 'exit' to quit.")

    while True:
        user_input = input("Agent: How can I help you? \nUser: ")
        if user_input.lower() == "exit":
            break

        # Create an instance of the orchestrator
        orchestrator = AnthropicOrchestrator(api_key="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")
        orchestrator.register_tool(EMAIL_TOOL_DEF, fetch_emails)
        # user_prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, find me the itinerary of my latest upcoming trip?"
        base_prompt = ""
        try:
            with open(BASE_PROMPT, 'r') as file:
                # Read the entire content of the file
                base_prompt = file.read()
        except FileNotFoundError:
            print("File not found. Please check the file path.")
        prompt = base_prompt.replace("{{user_prompt}}", user_input)
        final_msg = orchestrator.chat(prompt)
        print("Agent: " + final_msg.content[0].text + "\n")
