"""Tool definition (what the model sees) and its implementation ( what we run)"""

import datetime

import requests

# JSON schema in the OPENAI "tools" format. The model never runs the tool itself:
# it only returns the tool name and arguements, and the code executes the call.
# The description matters, because the model decides from it when to call the tool.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_calendar_day",
            "description": (
                "Reads all appointments of a single calendar day. "
                "To cover several days, call this tool once per day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in the format YYYY-MM-DD, e.g. 2026-10-29",
                    }
                },
                "required": ["date"],
            },
        },
    }
]


def get_calendar_day(date: str, base_url: str) -> dict:
    """Load the appontments of one day from the calendar endpoint.

    Errors are returned as {error: ...} instead of being raised, so the model can see
    what went wrong and esplains it to the user.
    """

    # validate the format first
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        return {"error": f"Invalid date '{date}', exprected format YYYY-MM-DD."}

    url = f"{base_url}/calendar/{date}.json"
    try:
        # timeout to prevent the agent to hang forever on a dead server
        resp = requests.get(url, timeout=10)
    except requests.RequestException as e:
        return {"error": f"Network error: {e}"}

    # there is one file per day, so if http 404 -> the day is not in the mock data
    if resp.status_code == 404:
        return {"error": f"No calendar data for {date} (outside the available range?)."}
    if not resp.ok:
        return {"error": f"HTTP {resp.status_code} while loading {date}."}

    # pass unchanged json to model
    return resp.json()
