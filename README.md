# agentic-interaction-ws2627

Repository for the exercises of the "Agentic Human-AI Interaction" lecture - WS2627

## Calendar Agent

A small command-line agent that answers questions about a calendar. It uses an LLM
(default: `openai/gpt-oss-20b` on Groq, via the OpenAI-compatible Chat Completions API)
with tool calling. The agent reads the appointments of a day from a read-only mock
calendar that is queried over HTTP.

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
```

### How it works

1. The task is sent to the model together with a system prompt and the tool definition.
2. The model calls the tool `get_calendar_day(date)` as often as needed (once per day).
3. The tool loads `{DATA_BASE_URL}/calendar/<YYYY-MM-DD>.json` over HTTP. The data is
   never copied locally.
4. The model answers based on the tool results.
The mock calendar covers 26 October 2026 to 8 November 2026. Requests outside this
range return an error to the model, which then tells the user that no data is available.

Currently the agent can only read appointments; it cannot create or change them.

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
│   └── tools.py       # tool definition and calendar access
├── logs/              # created at runtime (git-ignored)
├── .env               # local configuration (git-ignored)
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
