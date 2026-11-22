"""Tests for the agent loop (agent/agent.py) with a scripted fake model.

The fake model (see tests/helpers.py) returns prepared responses or raises
prepared API errors, so no network and no real model are needed.
"""

import re
from datetime import date
from types import SimpleNamespace

import openai
import pytest

from agent import agent as agent_module
from tests.helpers import (
    CFG,
    FakeCalendar,
    FakeTool,
    auth_error,
    connection_error,
    events,
    other_bad_request,
    parse_error,
    rate_limit,
    read_log,
    reply,
    server_error,
    tool_call,
)

# Matches the position part of an error detail, whatever wording surrounds it.
STEP_LABEL = re.compile(r"step \d+, attempt \d+/\d+")


def run(log, task="What is on 29 Oct 2026?"):
    return agent_module.run(task, CFG, log)


def entries_of(log, event):
    return [entry for entry in read_log(log.path) if entry["event"] == event]


# --- the normal path --------------------------------------------------------------


def test_a_plain_answer_needs_one_model_call(start_run):
    llm, log = start_run(reply("Nothing planned."))
    assert run(log) == "Nothing planned."
    assert len(llm.requests) == 1
    assert events(read_log(log.path)) == ["user_task", "final"]


def test_the_first_request_holds_the_system_prompt_and_the_task(start_run):
    llm, log = start_run(reply("ok"))
    run(log, task="Anything on Friday?")
    system, user = llm.requests[0]
    assert system["role"] == "system"
    assert user == {"role": "user", "content": "Anything on Friday?"}


def test_endpoint_model_key_and_tool_come_from_the_config(start_run):
    llm, log = start_run(reply("ok"))
    run(log)
    assert llm.client_kwargs["base_url"] == CFG["OPENAI_BASE_URL"]
    assert llm.client_kwargs["api_key"] == CFG["OPENAI_API_KEY"]
    assert llm.client_kwargs["timeout"] > 0
    (kwargs,) = llm.create_kwargs
    assert kwargs["model"] == CFG["MODEL"]
    assert [t["function"]["name"] for t in kwargs["tools"]] == [
        "get_calendar_day",
        "get_weather",
        "get_train_connections",
    ]


def test_a_two_day_task_runs_through_with_one_call(start_run):
    calendar = FakeCalendar(
        {"2026-10-29": {"events": ["Lecture"]}, "2026-10-30": {"events": ["Exam"]}}
    )
    _llm, log = start_run(
        reply(tool_calls=[tool_call({"date": "2026-10-29"}, "c1")]),
        reply(tool_calls=[tool_call({"date": "2026-10-30"}, "c2")]),
        reply("29 Oct: Lecture. 30 Oct: Exam."),
        calendar=calendar,
    )
    assert run(log) == "29 Oct: Lecture. 30 Oct: Exam."
    assert calendar.days == ["2026-10-29", "2026-10-30"]
    # The data URL from the config is what the tool is given.
    assert calendar.base_urls == [CFG["DATA_BASE_URL"]] * 2
    assert events(read_log(log.path)) == [
        "user_task",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final",
    ]


def test_a_task_with_two_different_tools_runs_through_with_one_call(start_run):
    weather = FakeTool({"condition": "rain"})
    transit = FakeTool({"connections": [{"departure": "08:02"}]})
    llm, log = start_run(
        reply(
            tool_calls=[
                tool_call(
                    {"origin": "ulm", "destination": "muenchen", "date": "2026-10-29"},
                    "t1",
                    name="get_train_connections",
                ),
                tool_call(
                    {"city": "muenchen", "date": "2026-10-29"},
                    "w1",
                    name="get_weather",
                ),
            ]
        ),
        reply("Train at 08:02, take an umbrella: rain."),
        weather=weather,
        transit=transit,
    )
    assert run(log) == "Train at 08:02, take an umbrella: rain."
    assert transit.calls == [("ulm", "muenchen", "2026-10-29")]
    assert weather.calls == [("muenchen", "2026-10-29")]
    results = {
        m["tool_call_id"]: m["content"] for m in llm.requests[1] if m["role"] == "tool"
    }
    assert "08:02" in results["t1"]
    assert "rain" in results["w1"]
    entries = read_log(log.path)
    assert [e["tool"] for e in entries if e["event"] == "tool_result"] == [
        "get_train_connections",
        "get_weather",
    ]
    assert events(entries)[-1] == "final"


def test_the_tools_get_the_data_url_from_the_config(start_run):
    seen = []

    def spy(name):
        def tool(*args):
            seen.append((name, args[-1]))
            return {}

        return tool

    _llm, log = start_run(
        reply(
            tool_calls=[
                tool_call({"city": "ulm", "date": "2026-10-29"}, "w", "get_weather"),
                tool_call(
                    {"origin": "ulm", "destination": "muenchen", "date": "2026-10-29"},
                    "t",
                    "get_train_connections",
                ),
            ]
        ),
        reply("done"),
        weather=spy("weather"),
        transit=spy("transit"),
    )
    run(log)
    url = CFG["DATA_BASE_URL"]
    assert seen == [("weather", url), ("transit", url)]


def test_missing_weather_data_goes_to_the_model_and_the_run_continues(start_run):
    error = {"error": "No data for weather in ulm on 2026-11-20 (only some days)."}
    llm, log = start_run(
        reply(
            tool_calls=[
                tool_call({"city": "ulm", "date": "2026-11-20"}, "w", "get_weather")
            ]
        ),
        reply("There is no weather data for that day."),
        weather=FakeTool(error),
    )
    assert run(log) == "There is no weather data for that day."
    (result,) = entries_of(log, "tool_result")
    assert result["ok"] is False
    assert "No data for weather" in llm.requests[1][-1]["content"]


def test_broken_arguments_of_the_new_tools_do_not_crash_the_run(start_run):
    # The real tools are used: they reject the empty values before any request.
    _llm, log = start_run(
        reply(tool_calls=[tool_call(raw="{", call_id="w", name="get_weather")]),
        reply(
            tool_calls=[tool_call(raw="[]", call_id="t", name="get_train_connections")]
        ),
        reply("Sorry, I could not read that."),
    )
    assert run(log) == "Sorry, I could not read that."
    assert [e["ok"] for e in entries_of(log, "tool_result")] == [False, False]


def test_the_history_grows_with_the_tool_request_and_its_result(start_run):
    calendar = FakeCalendar({"2026-10-29": {"events": ["Lecture"]}})
    llm, log = start_run(
        reply(tool_calls=[tool_call({"date": "2026-10-29"}, "c1")]),
        reply("done"),
        calendar=calendar,
    )
    run(log)
    assistant, tool_message = llm.requests[1][-2:]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "get_calendar_day"
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "c1"
    assert "Lecture" in tool_message["content"]


def test_several_tool_calls_in_one_response_are_all_answered(start_run):
    calendar = FakeCalendar()
    llm, log = start_run(
        reply(
            tool_calls=[
                tool_call({"date": "2026-10-29"}, "a"),
                tool_call({"date": "2026-10-30"}, "b"),
            ]
        ),
        reply("done"),
        calendar=calendar,
    )
    run(log)
    assert calendar.days == ["2026-10-29", "2026-10-30"]
    ids = [m["tool_call_id"] for m in llm.requests[1] if m["role"] == "tool"]
    assert ids == ["a", "b"]


def test_log_entries_describe_the_steps(start_run):
    _llm, log = start_run(
        reply(tool_calls=[tool_call({"date": "2026-10-29"})]),
        reply("The answer."),
        calendar=FakeCalendar(),
    )
    run(log)
    entries = read_log(log.path)
    assert {e["run_id"] for e in entries} == {log.run_id}
    assert {e["model"] for e in entries} == {"test-model"}
    (call,) = entries_of(log, "tool_call")
    assert call["tool"] == "get_calendar_day"
    assert call["args"] == {"date": "2026-10-29"}
    (final,) = entries_of(log, "final")
    assert final["detail"] == "The answer."
    for entry in entries_of(log, "tool_result") + [final]:
        assert isinstance(entry["duration_ms"], int) and entry["duration_ms"] >= 0


# --- errors inside a step do not abort the run -----------------------------------------


def test_a_tool_error_goes_to_the_model_and_the_run_continues(start_run):
    calendar = FakeCalendar(
        {"2026-11-20": {"error": "No calendar data for 2026-11-20"}}
    )
    llm, log = start_run(
        reply(tool_calls=[tool_call({"date": "2026-11-20"})]),
        reply("There is no data for that day."),
        calendar=calendar,
    )
    assert run(log) == "There is no data for that day."
    (result,) = entries_of(log, "tool_result")
    assert result["ok"] is False
    assert "No calendar data" in llm.requests[1][-1]["content"]


def test_an_unknown_tool_is_reported_back(start_run):
    calendar = FakeCalendar()
    llm, log = start_run(
        reply(tool_calls=[tool_call({}, name="delete_everything")]),
        reply("I cannot do that."),
        calendar=calendar,
    )
    assert run(log) == "I cannot do that."
    assert calendar.days == []
    (result,) = entries_of(log, "tool_result")
    assert result["ok"] is False
    assert result["tool"] == "delete_everything"
    assert "Unknown tool" in llm.requests[1][-1]["content"]


@pytest.mark.parametrize(
    "raw",
    ['{"date": ', "not json", "", '"2026-10-29"', "[]", "42", "null"],
    ids=["cut off", "not json", "empty", "bare string", "list", "number", "null"],
)
def test_broken_tool_arguments_do_not_crash_the_run(start_run, raw):
    # The real get_calendar_day is used here: it rejects the empty date
    # before any request is made (requests.get is blocked in tests anyway).
    llm, log = start_run(
        reply(tool_calls=[tool_call(raw=raw)]),
        reply("Sorry, I could not read the date."),
    )
    assert run(log) == "Sorry, I could not read the date."
    (result,) = entries_of(log, "tool_result")
    assert result["ok"] is False
    assert "Invalid date" in llm.requests[1][-1]["content"]


def test_custom_tool_calls_are_ignored_and_function_calls_still_run(start_run):
    custom = SimpleNamespace(
        id="x", type="custom", custom=SimpleNamespace(name="n", input="i")
    )
    calendar = FakeCalendar()
    llm, log = start_run(
        reply(tool_calls=[custom, tool_call({"date": "2026-10-29"}, "c1")]),
        reply("done"),
        calendar=calendar,
    )
    assert run(log) == "done"
    assert calendar.days == ["2026-10-29"]
    assert len([m for m in llm.requests[1] if m["role"] == "tool"]) == 1


# --- the step limit ------------------------------------------------------------------------


def test_reaching_the_step_limit_raises_instead_of_returning(start_run, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_STEPS", 3)
    llm, log = start_run(
        *[
            reply(tool_calls=[tool_call({"date": "2026-10-29"}, f"c{i}")])
            for i in range(3)
        ],
        calendar=FakeCalendar(),
    )
    with pytest.raises(RuntimeError, match="No final answer after 3 steps"):
        run(log)
    assert len(llm.requests) == 3  # a fourth call would have raised IndexError
    assert entries_of(log, "final") == []


def test_an_answer_on_the_last_allowed_step_is_a_success(start_run, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_STEPS", 3)
    _llm, log = start_run(
        reply(tool_calls=[tool_call({"date": "2026-10-29"}, "c1")]),
        reply(tool_calls=[tool_call({"date": "2026-10-30"}, "c2")]),
        reply("Both days checked."),
        calendar=FakeCalendar(),
    )
    assert run(log) == "Both days checked."


def test_the_step_limit_counts_model_calls_not_tool_calls(start_run, monkeypatch):
    monkeypatch.setattr(agent_module, "MAX_STEPS", 2)
    calendar = FakeCalendar()
    calls = [tool_call({"date": f"2026-10-{day}"}, f"c{day}") for day in range(26, 31)]
    _llm, log = start_run(
        reply(tool_calls=calls), reply("All five days."), calendar=calendar
    )
    assert run(log) == "All five days."
    assert len(calendar.days) == 5


# --- retries for model problems ------------------------------------------------------------


def test_unparsable_model_output_is_retried(start_run):
    llm, log = start_run(parse_error(), reply("Recovered."))
    assert run(log) == "Recovered."
    assert len(llm.requests) == 2
    (error,) = entries_of(log, "error")
    assert error["ok"] is False
    assert "step 1, attempt 1/3" in error["detail"]
    assert "output_parse_failed" in error["detail"]
    assert "request_id" in error["detail"]  # the request ID is logged for debugging


def test_unparsable_output_every_time_ends_in_an_error(start_run):
    llm, log = start_run(*[parse_error() for _ in range(3)])
    with pytest.raises(openai.BadRequestError):
        run(log)
    assert len(llm.requests) == 3
    assert (
        len(entries_of(log, "error")) == 2
    )  # the third failure is raised, not logged here


def test_other_bad_requests_are_not_retried(start_run):
    llm, log = start_run(other_bad_request())
    with pytest.raises(openai.BadRequestError):
        run(log)
    assert len(llm.requests) == 1
    assert entries_of(log, "error") == []


def test_failed_attempts_are_counted_per_step_not_per_run(start_run):
    llm, log = start_run(
        parse_error(),
        parse_error(),
        reply(tool_calls=[tool_call({"date": "2026-10-29"})]),
        parse_error(),
        parse_error(),
        reply("Done."),
        calendar=FakeCalendar(),
    )
    assert run(log) == "Done."
    assert len(llm.requests) == 6
    labels = [STEP_LABEL.findall(e["detail"])[0] for e in entries_of(log, "error")]
    assert labels == [
        "step 1, attempt 1/3",
        "step 1, attempt 2/3",
        "step 2, attempt 1/3",
        "step 2, attempt 2/3",
    ]


@pytest.mark.parametrize(
    "content", ["", "   \n", None], ids=["empty", "whitespace", "none"]
)
def test_an_empty_answer_is_retried(start_run, content):
    _llm, log = start_run(reply(content), reply("A real answer."))
    assert run(log) == "A real answer."
    (error,) = entries_of(log, "error")
    assert "empty response (finish_reason=stop)" in error["detail"]


def test_an_empty_answer_every_time_is_an_error_not_a_blank_success(start_run):
    _llm, log = start_run(*[reply("") for _ in range(3)])
    with pytest.raises(RuntimeError, match="No usable model response after 3 attempts"):
        run(log)
    assert entries_of(log, "final") == []


# --- retries for temporary API problems ---------------------------------------------------


@pytest.mark.parametrize(
    "make_error",
    [rate_limit, server_error, connection_error],
    ids=["429", "5xx", "connection"],
)
def test_a_temporary_api_problem_is_retried_after_waiting(
    start_run, sleeps, make_error
):
    error = make_error()
    _llm, log = start_run(error, reply("Back again."))
    assert run(log) == "Back again."
    assert sleeps == [agent_module.RETRY_WAIT_SECONDS]
    (logged,) = entries_of(log, "error")
    assert logged["ok"] is False
    assert type(error).__name__ in logged["detail"]


def test_the_waiting_time_grows_and_the_run_finally_gives_up(start_run, sleeps):
    llm, log = start_run(*[rate_limit() for _ in range(3)])
    with pytest.raises(openai.RateLimitError):
        run(log)
    wait = agent_module.RETRY_WAIT_SECONDS
    assert sleeps == [wait, 2 * wait]  # no waiting after the last attempt
    assert len(llm.requests) == 3


def test_an_invalid_api_key_is_not_retried(start_run, sleeps):
    llm, log = start_run(auth_error())
    with pytest.raises(openai.AuthenticationError):
        run(log)
    assert len(llm.requests) == 1
    assert sleeps == []


# --- the system prompt ----------------------------------------------------------------------


def test_the_prompt_contains_today_and_the_calendar_range():
    prompt = agent_module.build_system_prompt("2026-10-26", "2026-11-08")
    today = date.today()
    assert today.isoformat() in prompt
    assert today.strftime("%A") in prompt
    assert "The calendar covers 2026-10-26 to 2026-11-08." in prompt


def test_the_prompt_names_all_tools_and_forbids_guessing_missing_data():
    prompt = agent_module.build_system_prompt("2026-10-26", "2026-11-08")
    for name in ("get_calendar_day", "get_weather", "get_train_connections"):
        assert name in prompt
    assert "Never estimate or guess" in prompt
    assert "start in Ulm" in prompt


def test_the_prompt_lists_every_calendar_day_with_its_weekday():
    prompt = agent_module.build_system_prompt("2026-10-26", "2026-11-08")
    assert "Available days: Mon 2026-10-26, Tue 2026-10-27" in prompt
    assert "Sun 2026-11-01, Mon 2026-11-02" in prompt
    assert "Sun 2026-11-08." in prompt


@pytest.mark.parametrize(
    "start, end",
    [
        ("2026-11-08", "2026-10-26"),
        ("not-a-date", "2026-11-08"),
        ("2026-10-26", "2026-12-31"),
    ],
    ids=["reversed", "invalid date", "too long"],
)
def test_an_unusable_or_huge_range_gets_no_day_list(start, end):
    assert "Available days" not in agent_module.build_system_prompt(start, end)


@pytest.mark.parametrize(
    "start, end",
    [(None, None), ("2026-10-26", None), (None, "2026-11-08"), ("", "")],
)
def test_an_unknown_range_is_left_out_of_the_prompt(start, end):
    prompt = agent_module.build_system_prompt(start, end)
    assert "The calendar covers" not in prompt
    assert "None" not in prompt


def test_the_range_from_the_index_reaches_the_model(start_run):
    llm, log = start_run(reply("ok"), day_range=("2026-10-26", "2026-11-08"))
    run(log)
    assert "2026-10-26 to 2026-11-08" in llm.requests[0][0]["content"]


def test_the_agent_works_when_the_range_is_unknown(start_run):
    llm, log = start_run(reply("ok"), day_range=(None, None))
    assert run(log) == "ok"
    assert "The calendar covers" not in llm.requests[0][0]["content"]
