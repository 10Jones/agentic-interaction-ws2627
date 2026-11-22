"""Tool definition (what the model sees) and its implementation ( what we run)"""

import datetime
import re

import requests
from openai.types.chat import ChatCompletionToolParam

# JSON schema in the OPENAI "tools" format. The model never runs the tool itself:
# it only returns the tool name and arguments, and the code executes the call.
# The description matters, because the model decides from it when to call the tool.

TOOLS: list[ChatCompletionToolParam] = [
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Reads the weather forecast for one city on one day. "
                "Not every city and day has data. An error means there is no "
                "forecast for that city or day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City code in lowercase ASCII, e.g. 'ulm' or "
                            "'muenchen' (München)"
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in the format YYYY-MM-DD, e.g. 2026-10-29",
                    },
                },
                "required": ["city", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_train_connections",
            "description": (
                "Reads the train connections (departure and arrival times) from one "
                "city to another on one day. Not every route and day has data. "
+                "An error means there are no connections for that route or day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": (
                            "Departure city code in lowercase ASCII, e.g. 'ulm'"
                        ),
                    },
                    "destination": {
                        "type": "string",
                        "description": (
                            "Arrival city code in lowercase ASCII, e.g. 'ulm' or "
                            "'muenchen' (München)"
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in the format YYYY-MM-DD, e.g. 2026-10-29",
                    },
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
]

# City codes end up in the URL path, so only plain characters are accepted.
_CITY_CODE = re.compile(r"[a-z0-9_-]+")
_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def get_calendar_range(base_url: str) -> tuple[str | None, str | None]:
    """Read the first and last available day from the index at the data URL.

    Returns (None, None) if the index cannot be loaded, so the agent still
    works without knowing the range.
    """
    try:
        resp = requests.get(f"{base_url}/", timeout=10)
        resp.raise_for_status()
        days = resp.json()["days"]
        return days["first"], days["last"]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        return None, None


def get_calendar_day(date: str, base_url: str) -> dict:
    """Load the appointments of one day from the calendar endpoint.

    Errors are returned as {error: ...} instead of being raised, so the model can see
    what went wrong and explains it to the user.
    """

    # validate the format first
    try:
        datetime.date.fromisoformat(date)
    except (TypeError, ValueError):
        return {"error": f"Invalid date '{date}', expected format YYYY-MM-DD."}

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
    try:
        return resp.json()
    except ValueError:
        return {"error": f"Calendar data for {date} is not valid JSON."}


def _city_code(city) -> str | None:
    """Turn e.g. 'München ' into 'muenchen'. None if it is not a plain city code."""
    if not isinstance(city, str):
        return None
    code = city.strip().lower().translate(_UMLAUTS)
    return code if _CITY_CODE.fullmatch(code) else None


def _is_iso_date(date) -> bool:
    """True only for a real date written exactly as YYYY-MM-DD."""
    try:
        return datetime.date.fromisoformat(date).isoformat() == date
    except (TypeError, ValueError):
        return False


def _get_json(url: str, what: str) -> dict:
    """Load one JSON document. Every problem is returned as {error: ...}.

    `what` names the request in the messages, e.g. "weather in ulm on 2026-10-29".
    Weather and connections exist only for single days, so HTTP 404 is a normal
    result ("no data"), not a crash.
    """
    try:
        resp = requests.get(url, timeout=10)
    except requests.RequestException as e:
        return {"error": f"Network error: {e}"}
    if resp.status_code == 404:
        return {"error": f"No data for {what} (only some cities and days are covered)."}
    if not resp.ok:
        return {"error": f"HTTP {resp.status_code} while loading {what}."}
    try:
        return resp.json()
    except ValueError:
        return {"error": f"Data for {what} is not valid JSON."}


def get_weather(city: str, date: str, base_url: str) -> dict:
    """Load the weather forecast of one city and day."""
    code = _city_code(city)
    if code is None:
        return {"error": f"Invalid city '{city}', expected a code like 'ulm'."}
    if not _is_iso_date(date):
        return {"error": f"Invalid date '{date}', expected format YYYY-MM-DD."}
    return _get_json(
        f"{base_url}/weather/{code}/{date}.json", f"weather in {code} on {date}"
    )


def get_train_connections(
    origin: str, destination: str, date: str, base_url: str
) -> dict:
    """Load the train connections between two cities on one day."""
    start, end = _city_code(origin), _city_code(destination)
    if start is None or end is None:
        return {
            "error": f"Invalid city '{origin}' or '{destination}', "
            "expected codes like 'ulm'."
        }
    if not _is_iso_date(date):
        return {"error": f"Invalid date '{date}', expected format YYYY-MM-DD."}
    return _get_json(
        f"{base_url}/transit/{start}/{end}/{date}.json",
        f"train connections {start} -> {end} on {date}",
    )
