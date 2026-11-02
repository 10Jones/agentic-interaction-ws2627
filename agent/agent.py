"""The agent loop: send the task to the model, run the requested tools, repeat."""

import json
import time
from datetime import date

from openai import OpenAI

from .tools import TOOLS, get_calendar_day

# max number of steps the model is allowed to take, so a confused model does not loop and burn tokens
MAX_STEPS = 8


def build_system_prompt() -> str:
    # model does not have a date, so we need to provide a date for relative expressions like "tomorrow"
    today = date.today()
    return (
        "You are a calendar assistant. Answer questions only based on the data "
        "returned by the tool get_calendar_day. Never invent appointments.\n"
        f"Today is {today.isoformat()} ({today.strftime('%A')}). "
        "The calendar covers 2026-10-26 to 2026-11-08. "
        "Convert relative dates such as 'tomorrow' or 'next Friday' into concrete dates. "
        "For questions that span several days, call the tool once for each day. "
        "Keep answers short and reply in the language of the question. "
        "You can currently only read appointments, not create or change them."
    )


def run(task: str, cfg: dict, log) -> str:
    client = OpenAI(base_url=cfg["OPENAI_BASE_URL"], api_key=cfg["OPENAI_API_KEY"])
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": task},
    ]
    log.log("user_task", detail=task)

    for _ in range(MAX_STEPS):
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=cfg["MODEL"], messages=messages, tools=TOOLS
        )
        llm_ms = int((time.perf_counter() - t0) * 1000)
        msg = resp.choices[0].message

        # no tool call means the model has finished and this is the final answer
        if not msg.tool_calls:
            answer = msg.content or ""
            log.log("final", duration_ms=llm_ms, detail=answer)
            return answer

        # TODO(clarify)
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
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
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

            # tool_call_id links this result to teh request above
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # if model kept requesting tools without ever answering
    raise RuntimeError(f"No final answer after {MAX_STEPS} steps.")
