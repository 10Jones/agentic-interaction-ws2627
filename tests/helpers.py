"""Shared test helpers: a scripted fake model, fake responses and API error builders."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx2
import openai

# Configuration handed to the agent in tests (never read from a real .env file).
CFG = {
    "OPENAI_BASE_URL": "https://llm.test/v1",
    "MODEL": "test-model",
    "OPENAI_API_KEY": "test-key",
    "DATA_BASE_URL": "https://data.test/v1",
}

_REQUEST = httpx2.Request("POST", "https://llm.test/v1/chat/completions")


# --- fake model responses ----------------------------------------------------


def tool_call(args=None, call_id="call_1", name="get_calendar_day", raw=None):
    """A function tool call as the model would request it.

    `args` is serialized to JSON. Pass `raw` to send an arbitrary (even broken)
    arguments string instead.
    """
    arguments = raw if raw is not None else json.dumps(args or {})
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def reply(content=None, tool_calls=None, finish_reason=None):
    """A chat completion with one choice. Without tool calls it is a final answer."""
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


class ScriptedLLM:
    """Stands in for the OpenAI client.

    Every call to chat.completions.create() returns (or raises) the next item of
    the script and records what the agent sent.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []  # snapshot of the messages of every request
        self.create_kwargs = []  # all other arguments of every request
        self.client_kwargs = {}  # arguments the client was created with
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # The agent keeps appending to one list, so store a copy of each state.
        self.requests.append(copy.deepcopy(kwargs["messages"]))
        self.create_kwargs.append({k: v for k, v in kwargs.items() if k != "messages"})
        # IndexError here means: the agent asked the model more often than expected.
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeCalendar:
    """Stands in for tools.get_calendar_day and records how it was called."""

    def __init__(self, results=None):
        self.results = results or {}
        self.days = []
        self.base_urls = []

    def __call__(self, date, base_url):
        self.days.append(date)
        self.base_urls.append(base_url)
        return self.results.get(date, {"date": date, "events": []})


class FakeTool:
    """Stands in for get_weather / get_train_connections and records the calls.

    A call is stored as the tuple of its arguments without the base URL.
    """

    def __init__(self, result=None):
        self.result = result if result is not None else {}
        self.calls = []

    def __call__(self, *args):
        *tool_args, _base_url = args
        self.calls.append(tuple(tool_args))
        return self.result


# --- API errors ----------------------------------------------------------------


def _http_response(status):
    return httpx2.Response(status, request=_REQUEST)


def parse_error():
    """400 that Groq sends when the model produced an unparsable tool call."""
    body = {
        "message": "Parsing failed. The model generated output that could not be parsed.",
        "type": "invalid_request_error",
        "code": "output_parse_failed",
        "failed_generation": "",
    }
    return openai.BadRequestError(
        "Parsing failed.", response=_http_response(400), body=body
    )


def other_bad_request():
    """A 400 that is not a parsing problem and will not fix itself."""
    return openai.BadRequestError(
        "model not found",
        response=_http_response(400),
        body={"message": "model not found"},
    )


def rate_limit():
    return openai.RateLimitError("rate limit", response=_http_response(429), body=None)


def server_error():
    return openai.InternalServerError(
        "overloaded", response=_http_response(500), body=None
    )


def connection_error():
    return openai.APIConnectionError(request=_REQUEST)


def auth_error():
    return openai.AuthenticationError(
        "bad key", response=_http_response(401), body=None
    )


# --- log helpers ---------------------------------------------------------------


def read_log(path):
    """Return the entries of a JSON Lines log file as a list of dicts."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def events(entries):
    """The event names of log entries, in order."""
    return [entry["event"] for entry in entries]
