"""The agent loop: send the task to the model, run the requested tools, repeat."""

import json
import time
from datetime import date, timedelta

from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .tools import (
    TOOLS,
    get_calendar_day,
    get_calendar_range,
    get_train_connections,
    get_weather,
)

# max number of steps the model is allowed to take, so a confused model does not loop and burn tokens
# The model usually requests one day per round, so a whole 14-day calendar needs about 15 rounds.
MAX_STEPS = 20

# Attempts per model request. The model sometimes produces a malformed tool call,
# which the API reports as HTTP 400 ("failed_generation"). This is random, so a
# retry usually works.
MAX_RETRIES = 3

# Base wait between retries after a temporary API problem (multiplied by the
# attempt number). The free tier allows only 8,000 tokens per minute, so a
# rate limit (429) needs some time to clear.
RETRY_WAIT_SECONDS = 10


def list_days(start: str | None, end: str | None) -> str:
    """One entry per calendar day ("Mon 2026-10-26, Tue ..."), so the model does
    not have to work out dates and weekdays itself. Empty if the range is unusable."""
    if not start or not end:
        return ""
    try:
        first, last = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        return ""
    count = (last - first).days + 1
    if not 0 < count <= 62:  # reversed range, or too long for a prompt
        return ""
    days = (first + timedelta(days=i) for i in range(count))
    return ", ".join(f"{d.strftime('%a')} {d.isoformat()}" for d in days)


def build_system_prompt(
    start_date_calendar: str | None, end_date_calendar: str | None
) -> str:
    # model does not have a date, so we need to provide a date for relative expressions like "tomorrow"
    today = date.today()
    # Only mention the range if it is known (the index may be unreachable).
    coverage = (
        f"The calendar covers {start_date_calendar} to {end_date_calendar}. "
        if start_date_calendar and end_date_calendar
        else ""
    )
    days = list_days(start_date_calendar, end_date_calendar)
    if coverage and days:
        coverage += f"Available days: {days}. "

    return (
        "You are a calendar and travel assistant. Answer questions only based on "
        "the data returned by your tools. Never invent appointments, weather "
        "values or train times.\n"
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        f"{coverage}"
        "Convert relative dates such as 'tomorrow', 'next Friday' or 'last week' "
        "into concrete dates. Weeks run from Monday to Sunday: 'this week' is the week "
        "that contains today, 'last week' is the week before it. "
        "get_calendar_day reads one day per call. If a question needs several days, "
        "request the tool once for each day. "
        "get_weather gives the forecast for one city and day, get_train_connections "
        "the trains between two cities on one day. City codes are lowercase ASCII, "
        "for example 'ulm' or 'muenchen' (München). Not every city, route and day "
        "has weather or train data. "
        "If the user names no starting point for a trip, start in Ulm. For a trip, "
        "the weather matters at the starting point and at the destination, because "
        "the user has to leave from there too: call get_weather for both cities on "
        "the travel day and mention both in the answer. "
        "If a task needs several tools, use all of them and name the results of each "
        "in your answer. If a tool returns an error, tell the user that there is no "
        "data for that place or day. Never estimate or guess the missing values. "
        # "all of them in the same response (several tool calls at once) instead of "
        # "them in the same response (several tool calls at once) instead of one after another. "
        "Keep answers short and reply in the language of the question. "
        "You can only read data, not create or change appointments."
    )


def create_with_retry(
    client: OpenAI,
    cfg: dict,
    messages: list[ChatCompletionMessageParam],
    log,
    step: int,
) -> ChatCompletion:
    """Send the request; retry on bad output, temporary API errors or empty answers."""
    for attempt in range(1, MAX_RETRIES + 1):
        # Identifies the failed request in the log (attempts restart at 1 per step).
        where = f"Model step {step}, attempt {attempt}/{MAX_RETRIES}"
        try:
            resp = client.chat.completions.create(
                model=cfg["MODEL"], messages=messages, tools=TOOLS
            )
        except BadRequestError as e:
            # Only retry the "model produced unparsable output" case. Other 400s
            # (wrong model name, invalid request) will not fix themselves.
            if "failed_generation" not in str(e.body) or attempt == MAX_RETRIES:
                raise
            # json.dumps keeps the detail valid JSON instead of a Python dict repr.
            body = json.dumps(e.body, ensure_ascii=False, default=str)
            log.log(
                "error", ok=False, detail=f"{where} (request_id={e.request_id}): {body}"
            )
            continue
        except (RateLimitError, APIConnectionError, InternalServerError) as e:
            # Temporary problems: rate limit (429), network or timeout, server error
            # Wait and try again instead of aborting the whole run.
            if attempt == MAX_RETRIES:
                raise
            log.log("error", ok=False, detail=f"{where}: {type(e).__name__}: {e}")
            time.sleep(RETRY_WAIT_SECONDS * attempt)
            continue

        # A usable response contains either a tool call or some text.
        choice = resp.choices[0]
        if choice.message.tool_calls or (choice.message.content or "").strip():
            return resp
        log.log(
            "error",
            ok=False,
            detail=f"{where}: empty response (finish_reason={choice.finish_reason})",
        )

    raise RuntimeError(f"No usable model response after {MAX_RETRIES} attempts.")


def run(task: str, cfg: dict, log) -> str:
    # The timeout keeps a hanging connection from blocking the run for minutes
    # (the SDK default is 10 minutes).
    client = OpenAI(
        base_url=cfg["OPENAI_BASE_URL"],
        api_key=cfg["OPENAI_API_KEY"],
        timeout=60.0,
    )

    # The available day range comes from the data index, so it is never hardcoded.
    start_date_calendar, end_date_calendar = get_calendar_range(cfg["DATA_BASE_URL"])

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": build_system_prompt(start_date_calendar, end_date_calendar),
        },
        {"role": "user", "content": task},
    ]
    log.log("user_task", detail=task)

    for step in range(1, MAX_STEPS + 1):
        t0 = time.perf_counter()
        resp = create_with_retry(client, cfg, messages, log, step)
        llm_ms = int((time.perf_counter() - t0) * 1000)
        msg = resp.choices[0].message

        # no tool call means the model has finished and this is the final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            log.log("final", duration_ms=llm_ms, detail=answer)
            return answer

        # The SDK type also allows "custom" tool calls, which have no .function.
        # We only define function tools, so keep just those.
        tool_calls = [tc for tc in msg.tool_calls if tc.type == "function"]

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            # Valid JSON is not always an object (e.g. a bare string).
            if not isinstance(args, dict):
                args = {}
            log.log("tool_call", tool=name, args=args, duration_ms=llm_ms)

            t1 = time.perf_counter()
            if name == "get_calendar_day":
                result = get_calendar_day(args.get("date", ""), cfg["DATA_BASE_URL"])
            elif name == "get_weather":
                result = get_weather(
                    args.get("city", ""), args.get("date", ""), cfg["DATA_BASE_URL"]
                )
            elif name == "get_train_connections":
                result = get_train_connections(
                    args.get("origin", ""),
                    args.get("destination", ""),
                    args.get("date", ""),
                    cfg["DATA_BASE_URL"],
                )
            else:
                # report unknown tools back to the model instead of crashing
                result = {"error": f"Unknown tool: {name}"}
            tool_ms = int((time.perf_counter() - t1) * 1000)

            log.log(
                "tool_result",
                tool=name,
                args=args,
                ok="error" not in result,
                duration_ms=tool_ms,
            )

            # tool_call_id links this result to the request above
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # if model kept requesting tools without ever answering
    raise RuntimeError(f"No final answer after {MAX_STEPS} steps.")
