"""Tests for agent/tools.py. All HTTP traffic is faked."""

import pytest
import requests

from agent import tools

BASE = "https://data.test/v1"


class FakeResponse:
    """The parts of requests.Response that the tools use."""

    def __init__(self, status_code=200, data=None, bad_json=False):
        self.status_code = status_code
        self._data = data
        self._bad_json = bad_json

    @property
    def ok(self):
        return self.status_code < 400

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        if self._bad_json:
            raise ValueError("not json")
        return self._data


class Http:
    """Replaces requests.get. Set `outcome` to a response or an exception to raise."""

    def __init__(self, monkeypatch):
        self.calls = []
        self.outcome = FakeResponse(200, {})
        monkeypatch.setattr(requests, "get", self._get)

    def _get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def http(monkeypatch):
    return Http(monkeypatch)


# --- tool definition ---------------------------------------------------------


def test_the_tool_schema_describes_three_functions_with_required_arguments():
    required = {
        t["function"]["name"]: t["function"]["parameters"]["required"]
        for t in tools.TOOLS
    }
    assert all(t["type"] == "function" for t in tools.TOOLS)
    assert required == {
        "get_calendar_day": ["date"],
        "get_weather": ["city", "date"],
        "get_train_connections": ["origin", "destination", "date"],
    }


# --- get_calendar_day -----------------------------------------------------------


def test_returns_the_json_of_the_day_unchanged(http):
    data = {"date": "2026-10-29", "events": [{"title": "Lecture", "start": "10:15"}]}
    http.outcome = FakeResponse(200, data)
    assert tools.get_calendar_day("2026-10-29", BASE) == data


def test_requests_the_file_of_that_day_with_a_timeout(http):
    http.outcome = FakeResponse(200, {})
    tools.get_calendar_day("2026-10-29", BASE)
    ((url, kwargs),) = http.calls
    assert url == f"{BASE}/calendar/2026-10-29.json"
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "data", [{}, {"events": []}, []], ids=["{}", "no events", "[]"]
)
def test_a_day_without_events_is_not_an_error(http, data):
    http.outcome = FakeResponse(200, data)
    assert tools.get_calendar_day("2026-10-31", BASE) == data


@pytest.mark.parametrize(
    "bad_date",
    [
        "",
        "tomorrow",
        "2026-13-01",
        "2026-02-30",
        "29.10.2026",
        "../../secret",
        "2026-10-29/../../x",
        "2026-10-29.json?x=1",
    ],
)
def test_an_invalid_date_is_rejected_without_a_request(http, bad_date):
    result = tools.get_calendar_day(bad_date, BASE)
    assert "Invalid date" in result["error"]
    assert http.calls == []


@pytest.mark.parametrize("bad_date", [None, 20261029, ["2026-10-29"], 1.5])
def test_a_non_string_date_is_an_error_not_a_crash(http, bad_date):
    result = tools.get_calendar_day(bad_date, BASE)
    assert "error" in result
    assert http.calls == []


def test_a_missing_day_is_reported_as_no_data(http):
    http.outcome = FakeResponse(404)
    result = tools.get_calendar_day("2026-11-20", BASE)
    assert "No calendar data for 2026-11-20" in result["error"]


@pytest.mark.parametrize("status", [400, 403, 500, 503])
def test_other_http_errors_name_the_status(http, status):
    http.outcome = FakeResponse(status)
    result = tools.get_calendar_day("2026-10-29", BASE)
    assert str(status) in result["error"]


@pytest.mark.parametrize(
    "error",
    [
        requests.ConnectionError("boom"),
        requests.Timeout("too slow"),
        requests.exceptions.SSLError("bad certificate"),
    ],
    ids=["connection", "timeout", "ssl"],
)
def test_network_problems_are_returned_as_errors(http, error):
    http.outcome = error
    result = tools.get_calendar_day("2026-10-29", BASE)
    assert result["error"].startswith("Network error")


def test_an_answer_that_is_not_json_is_an_error_not_a_crash(http):
    http.outcome = FakeResponse(200, bad_json=True)
    result = tools.get_calendar_day("2026-10-29", BASE)
    assert "not valid JSON" in result["error"]


# --- get_calendar_range -----------------------------------------------------------


def test_the_range_comes_from_the_index(http):
    http.outcome = FakeResponse(
        200, {"days": {"first": "2026-10-26", "last": "2026-11-08"}, "version": "v1"}
    )
    assert tools.get_calendar_range(BASE) == ("2026-10-26", "2026-11-08")


def test_the_index_is_requested_with_a_trailing_slash(http):
    http.outcome = FakeResponse(200, {"days": {"first": "a", "last": "b"}})
    tools.get_calendar_range(BASE)
    ((url, kwargs),) = http.calls
    assert url == f"{BASE}/"
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "outcome",
    [
        FakeResponse(404),
        FakeResponse(500),
        FakeResponse(200, bad_json=True),
        FakeResponse(200, {}),
        FakeResponse(200, {"days": {}}),
        FakeResponse(200, {"days": {"first": "2026-10-26"}}),
        FakeResponse(200, {"days": None}),
        FakeResponse(200, {"days": []}),
        FakeResponse(200, []),
        requests.ConnectionError("boom"),
        requests.Timeout("too slow"),
    ],
    ids=[
        "404",
        "500",
        "not json",
        "no days key",
        "empty days",
        "last missing",
        "days is null",
        "days is a list",
        "index is a list",
        "connection error",
        "timeout",
    ],
)
def test_an_unusable_index_gives_an_unknown_range_instead_of_a_crash(http, outcome):
    http.outcome = outcome
    assert tools.get_calendar_range(BASE) == (None, None)


# --- get_weather ------------------------------------------------------------------


def test_the_weather_json_is_returned_unchanged(http):
    data = {"city": "muenchen", "date": "2026-10-29", "condition": "rain"}
    http.outcome = FakeResponse(200, data)
    assert tools.get_weather("muenchen", "2026-10-29", BASE) == data


def test_the_weather_file_of_that_city_and_day_is_requested_with_a_timeout(http):
    http.outcome = FakeResponse(200, {})
    tools.get_weather("ulm", "2026-10-29", BASE)
    ((url, kwargs),) = http.calls
    assert url == f"{BASE}/weather/ulm/2026-10-29.json"
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize("city", ["München", " Muenchen ", "MUENCHEN"])
def test_spellings_of_a_city_are_mapped_to_its_code(http, city):
    http.outcome = FakeResponse(200, {})
    tools.get_weather(city, "2026-10-29", BASE)
    assert http.calls[0][0] == f"{BASE}/weather/muenchen/2026-10-29.json"


@pytest.mark.parametrize(
    "bad_city", ["", "../../secret", "ulm/../x", "ulm?x=1", None, 5, ["ulm"]]
)
def test_an_invalid_city_is_rejected_without_a_request(http, bad_city):
    result = tools.get_weather(bad_city, "2026-10-29", BASE)
    assert "Invalid city" in result["error"]
    assert http.calls == []


@pytest.mark.parametrize(
    "bad_date", ["", "tomorrow", "2026-02-30", "20261029", "../x", None, 20261029]
)
def test_an_invalid_weather_date_is_rejected_without_a_request(http, bad_date):
    result = tools.get_weather("ulm", bad_date, BASE)
    assert "Invalid date" in result["error"]
    assert http.calls == []


def test_no_weather_for_a_day_is_reported_as_no_data(http):
    http.outcome = FakeResponse(404)
    result = tools.get_weather("ulm", "2026-11-20", BASE)
    assert "No data for weather in ulm on 2026-11-20" in result["error"]


def test_no_weather_for_an_unknown_city_is_reported_as_no_data(http):
    http.outcome = FakeResponse(404)
    result = tools.get_weather("berlin", "2026-10-29", BASE)
    assert "No data for weather in berlin" in result["error"]


@pytest.mark.parametrize("status", [400, 500, 503])
def test_other_weather_http_errors_name_the_status(http, status):
    http.outcome = FakeResponse(status)
    assert str(status) in tools.get_weather("ulm", "2026-10-29", BASE)["error"]


def test_a_weather_network_problem_is_returned_as_an_error(http):
    http.outcome = requests.Timeout("too slow")
    result = tools.get_weather("ulm", "2026-10-29", BASE)
    assert result["error"].startswith("Network error")


def test_weather_that_is_not_json_is_an_error_not_a_crash(http):
    http.outcome = FakeResponse(200, bad_json=True)
    assert "not valid JSON" in tools.get_weather("ulm", "2026-10-29", BASE)["error"]


# --- get_train_connections ----------------------------------------------------------


def test_the_connections_json_is_returned_unchanged(http):
    data = {"connections": [{"departure": "08:02", "arrival": "09:35"}]}
    http.outcome = FakeResponse(200, data)
    assert tools.get_train_connections("ulm", "muenchen", "2026-10-29", BASE) == data


def test_the_connections_file_of_that_route_and_day_is_requested(http):
    http.outcome = FakeResponse(200, {})
    tools.get_train_connections("Ulm", "München", "2026-10-29", BASE)
    ((url, kwargs),) = http.calls
    assert url == f"{BASE}/transit/ulm/muenchen/2026-10-29.json"
    assert kwargs["timeout"] > 0


@pytest.mark.parametrize(
    "origin, destination",
    [("", "ulm"), ("ulm", ""), ("../x", "ulm"), ("ulm", "a/b"), (None, "ulm")],
)
def test_an_invalid_city_in_a_route_is_rejected_without_a_request(
    http, origin, destination
):
    result = tools.get_train_connections(origin, destination, "2026-10-29", BASE)
    assert "Invalid city" in result["error"]
    assert http.calls == []


def test_an_invalid_connections_date_is_rejected_without_a_request(http):
    result = tools.get_train_connections("ulm", "muenchen", "soon", BASE)
    assert "Invalid date" in result["error"]
    assert http.calls == []


def test_no_connections_for_a_day_are_reported_as_no_data(http):
    http.outcome = FakeResponse(404)
    result = tools.get_train_connections("ulm", "muenchen", "2026-11-20", BASE)
    assert "No data for train connections ulm -> muenchen" in result["error"]


def test_a_connections_network_problem_is_returned_as_an_error(http):
    http.outcome = requests.ConnectionError("boom")
    result = tools.get_train_connections("ulm", "muenchen", "2026-10-29", BASE)
    assert result["error"].startswith("Network error")
