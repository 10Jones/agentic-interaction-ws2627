# agentic-interaction-ws2627

Repository for the exercises of the "Agentic Human-AI Interaction" lecture - WS2627

## Calendar Agent

A small command-line agent that answers questions about a calendar, the weather and
train connections. It uses an LLM (default: `openai/gpt-oss-20b` on Groq, via the
OpenAI-compatible Chat Completions API) with tool calling. The agent reads all data
from read-only mock data that is queried over HTTP.

### Requirements

- Python 3.15
- A Groq API key (free tier is sufficient)
- Network access to the LLM endpoint and the mock data repository

### Setup

1. Create and activate a virtual environment in the repository root.

   Windows (cmd):

```cmd
   python -m venv .venv
   .venv\Scripts\activate
```

   Windows (PowerShell): `.venv\Scripts\Activate.ps1`

   Linux / macOS:

```cmd
   python3 -m venv .venv
   source .venv/bin/activate
```

2. Install the dependencies:

```cmd
   pip install -r requirements.txt
```

3. Create a `.env` file in the repository root. You can copy `.env.example` and fill in the values:

```dotenv
   OPENAI_BASE_URL=https://api.groq.com/openai/v1
   MODEL=openai/gpt-oss-20b
   OPENAI_API_KEY=<your Groq API key>
   DATA_BASE_URL=<base URL of the mock data repository, without trailing slash>
```

   | Variable          | Description                                                     |
   | ----------------- | --------------------------------------------------------------- |
   | `OPENAI_BASE_URL` | OpenAI-compatible endpoint of the model backend                 |
   | `MODEL`           | Model name as sent to the endpoint                              |
   | `OPENAI_API_KEY`  | API key for the backend                                         |
   | `DATA_BASE_URL`   | Base URL of the mock data (calendar files are loaded from here) |

   The `.env` file is listed in `.gitignore` and must not be committed.

### Usage

Run the agent from the repository root with the task in quotes:

```cmd
python -m agent "What appointments do I have on 29 October 2026?"
```

The agent prints its answer to stdout and exits with code `0`. On errors it prints a
message to stderr and exits with a non-zero code.

More examples:

```cmd
python -m agent "When is my first appointment on 2026-11-03?"
python -m agent "Which days between 26 and 30 October have no appointments?"
python -m agent "How do I get to Munich on 29 Oct 2026, and do I need an umbrella?"
```

### How it works

1. The agent reads the available day range from the data index.
2. The task is sent to the model together with a system prompt and the tool definition.
3. The model calls the tools as often as needed, also several different tools in one
   run.
4. Each tool loads one JSON file over HTTP. The data is never copied locally.
5. The model answers based on the tool results.

| Tool                                                | Loads                                              |
| --------------------------------------------------- | -------------------------------------------------- |
| `get_calendar_day(date)`                            | `{DATA_BASE_URL}/calendar/<date>.json`             |
| `get_weather(city, date)`                           | `{DATA_BASE_URL}/weather/<city>/<date>.json`       |
| `get_train_connections(origin, destination, date)`  | `{DATA_BASE_URL}/transit/<origin>/<destination>/<date>.json` |

Dates are `YYYY-MM-DD`, cities are the codes from the index (`ulm`, `muenchen`).
Spellings like "München" are mapped to the code. If no starting point is named for a
trip, the system prompt tells the model to start in Ulm.

Weather and connections exist only for some days. An HTTP 404 is returned to the model
as `{"error": "No data for ..."}`, and the system prompt tells it to say that no data is
available instead of estimating values.

The available day range is read at start-up from the index at `DATA_BASE_URL`
(`days.first` and `days.last`) and passed to the model in the system prompt. If the
index cannot be loaded, the agent still works without knowing the range. Requests for
days outside the calendar return an error to the model, which then tells the user that
no data is available.

Currently the agent can only read data; it cannot create or change appointments.

### Logging

Every run writes a log file to `logs/` (`<timestamp>_<run_id>.jsonl`, one JSON object per line).

| Field         | Content                                                              |
| ------------- | -------------------------------------------------------------------- |
| `ts`          | Timestamp, ISO 8601 (UTC)                                            |
| `run_id`      | One ID per run                                                       |
| `event`       | `user_task`, `tool_call`, `tool_result`, `final` or `error`          |
| `tool`        | Tool name (for tool events)                                          |
| `args`        | Tool arguments (for tool events)                                     |
| `ok`          | Whether the step succeeded                                           |
| `model`       | Model name as sent to the endpoint                                   |
| `duration_ms` | Duration of the step in milliseconds                                 |
| `detail`      | Optional extra information (task text, final answer or error message)|

For `tool_call` and `final`, `duration_ms` is the duration of the model request;
for `tool_result` it is the duration of the tool execution.

### Project structure

```list
.
├── agent/
│   ├── __init__.py
│   ├── __main__.py    # entry point (python -m agent)
│   ├── agent.py       # agent loop and prompt
│   ├── config.py      # reads the environment / .env
│   ├── logger.py      # JSON Lines run log
│   └── tools.py       # tool definitions, calendar, weather and train access
├── logs/              # created at runtime (git-ignored)
├── .env               # local configuration (git-ignored)
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
