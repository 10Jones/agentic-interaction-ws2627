"""The agent loop: send the task to the model, run the requested tools, repeat."""

import json
import time
from datetime import date

from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam

from .tools import TOOLS, get_calendar_day, get_calendar_range

# max number of steps the model is allowed to take, so a confused model does not loop and burn tokens
# The model usually requests one day per round, so a whole 14-day calendar needs about 15 rounds.
MAX_STEPS = 20

# Attempts per model request. The model sometimes produces a malformed tool call,
# which the API reports as HTTP 400 ("failed_generation"). This is random, so a
# retry usually works.
MAX_RETRIES = 3


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
    return (
        "You are a calendar assistant. Answer questions only based on the data "
        "returned by the tool get_calendar_day. Never invent appointments.\n"
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        f"{coverage}"
        "Convert relative dates such as 'tomorrow', 'next Friday' or 'last week' "
        "into concrete dates (weeks run from Monday to Sunday). "
        "For questions that span several days, call the tool once for each day. "
        "Keep answers short and reply in the language of the question. "
        "You can currently only read appointments, not create or change them."
    )


def create_with_retry(
    client: OpenAI,
    cfg: dict,
    messages: list[ChatCompletionMessageParam],
    log,
    step: int,
) -> ChatCompletion:
    """Send the request; retry on unparsable output or an empty answer."""
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
            log.log("error", ok=False, detail=f"{where}: {body}")
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
    client = OpenAI(base_url=cfg["OPENAI_BASE_URL"], api_key=cfg["OPENAI_API_KEY"])

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
            log.log("tool_call", tool=name, args=args, duration_ms=llm_ms)

            t1 = time.perf_counter()
            if name == "get_calendar_day":
                result = get_calendar_day(args.get("date", ""), cfg["DATA_BASE_URL"])
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
