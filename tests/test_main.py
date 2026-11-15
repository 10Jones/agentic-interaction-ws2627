"""Tests for the entry point (agent/__main__.py): exit codes, output and log."""

import sys

import pytest

from agent import __main__ as cli
from agent import agent as agent_module
from agent import tools
from agent.logger import RunLogger
from tests.helpers import (
    CFG,
    FakeCalendar,
    ScriptedLLM,
    auth_error,
    events,
    rate_limit,
    read_log,
    reply,
    tool_call,
)


@pytest.fixture
def created_loggers(monkeypatch, tmp_path):
    """Run main() in a temporary directory with a fixed config.

    Returns the list of loggers that main() created, so tests can inspect them.
    """
    monkeypatch.chdir(tmp_path)  # the log directory "logs" is created relative to here
    monkeypatch.setattr(cli, "load_config", lambda: dict(CFG))
    created = []

    class SpyLogger(RunLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(cli, "RunLogger", SpyLogger)
    return created


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["agent", *args])
    return cli.main()


def script_the_model(monkeypatch, *script):
    """Let the real agent loop run against a scripted model."""
    llm = ScriptedLLM(*script)
    monkeypatch.setattr(agent_module, "OpenAI", lambda **kwargs: llm)
    monkeypatch.setattr(
        agent_module, "get_calendar_range", lambda base_url: (None, None)
    )
    monkeypatch.setattr(agent_module, "get_calendar_day", FakeCalendar())
    return llm


# --- arguments and configuration --------------------------------------------------------------


@pytest.mark.parametrize("args", [(), ("one", "two")], ids=["no task", "two arguments"])
def test_wrong_arguments_print_usage_and_return_2(
    monkeypatch, created_loggers, capsys, args
):
    assert run_cli(monkeypatch, *args) == 2
    assert 'python -m agent "<task>"' in capsys.readouterr().err
    assert created_loggers == []  # no run, no log


def test_missing_configuration_returns_1_with_a_message(
    monkeypatch, created_loggers, capsys
):
    def failing():
        raise RuntimeError("Missing environment variables: MODEL")

    monkeypatch.setattr(cli, "load_config", failing)
    assert run_cli(monkeypatch, "task") == 1
    assert "Missing environment variables: MODEL" in capsys.readouterr().err
    assert created_loggers == []  # the logger needs the model name from the config


# --- results of a run ----------------------------------------------------------------------------


def test_the_task_is_passed_on_and_the_answer_is_printed_with_exit_code_0(
    monkeypatch, created_loggers, capsys
):
    seen = {}

    def fake_run(task, cfg, log):
        seen["task"] = task
        return "Prüfung um 10 Uhr"

    monkeypatch.setattr(cli, "run", fake_run)
    assert run_cli(monkeypatch, "What is on Friday?") == 0
    assert seen["task"] == "What is on Friday?"
    assert capsys.readouterr().out == "Prüfung um 10 Uhr\n"


@pytest.mark.parametrize(
    "make_error",
    [lambda: RuntimeError("No final answer after 20 steps."), rate_limit, auth_error],
    ids=["step limit", "rate limit", "invalid key"],
)
def test_expected_failures_return_1_and_are_logged(
    monkeypatch, created_loggers, capsys, make_error
):
    error = make_error()

    def failing_run(task, cfg, log):
        raise error

    monkeypatch.setattr(cli, "run", failing_run)
    assert run_cli(monkeypatch, "task") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Error:")
    (log,) = created_loggers
    (entry,) = read_log(log.path)
    assert entry["event"] == "error"
    assert entry["ok"] is False
    assert str(error) in entry["detail"]


def test_an_unexpected_bug_is_logged_and_still_raised(monkeypatch, created_loggers):
    def buggy_run(task, cfg, log):
        raise ValueError("boom")

    monkeypatch.setattr(cli, "run", buggy_run)
    with pytest.raises(ValueError, match="boom"):  # Python then exits with code 1
        run_cli(monkeypatch, "task")
    (log,) = created_loggers
    (entry,) = read_log(log.path)
    assert entry["event"] == "error"
    assert entry["ok"] is False
    assert entry["detail"] == "unexpected ValueError: boom"


@pytest.mark.parametrize("fails", [False, True], ids=["success", "failure"])
def test_the_log_file_is_closed_in_every_case(monkeypatch, created_loggers, fails):
    def fake_run(task, cfg, log):
        if fails:
            raise RuntimeError("failed")
        return "answer"

    monkeypatch.setattr(cli, "run", fake_run)
    run_cli(monkeypatch, "task")
    (log,) = created_loggers
    assert log._file.closed  # private attribute, checked on purpose


# --- whole runs through the real agent loop ----------------------------------------------------


def test_a_multi_step_run_ends_with_an_answer_exit_code_0_and_a_final_entry(
    monkeypatch, created_loggers, capsys
):
    script_the_model(
        monkeypatch,
        reply(tool_calls=[tool_call({"date": "2026-10-29"}, "c1")]),
        reply(tool_calls=[tool_call({"date": "2026-10-30"}, "c2")]),
        reply("Both days are free."),
    )
    assert run_cli(monkeypatch, "What is on 29 and 30 Oct 2026?") == 0
    assert capsys.readouterr().out == "Both days are free.\n"
    (log,) = created_loggers
    entries = read_log(log.path)
    assert events(entries)[-1] == "final"
    assert events(entries).count("tool_call") == 2
    assert all(entry["ok"] for entry in entries)


def test_hitting_the_step_limit_ends_as_an_error_not_as_a_success(
    monkeypatch, created_loggers, capsys
):
    monkeypatch.setattr(agent_module, "MAX_STEPS", 2)
    script_the_model(
        monkeypatch,
        *[
            reply(tool_calls=[tool_call({"date": "2026-10-29"}, f"c{i}")])
            for i in range(2)
        ],
    )
    assert run_cli(monkeypatch, "loop forever") == 1
    assert capsys.readouterr().out == ""  # no answer is printed
    (log,) = created_loggers
    entries = read_log(log.path)
    assert "final" not in events(entries)
    assert entries[-1]["event"] == "error"
    assert entries[-1]["ok"] is False
    assert "No final answer after 2 steps" in entries[-1]["detail"]


def test_a_failing_tool_step_does_not_abort_the_run(
    monkeypatch, created_loggers, capsys
):
    script_the_model(
        monkeypatch,
        reply(tool_calls=[tool_call({"date": "not-a-date"})]),
        reply("That is not a valid date."),
    )
    # Use the real tool again: it rejects the invalid date without any request.
    monkeypatch.setattr(agent_module, "get_calendar_day", tools.get_calendar_day)
    assert run_cli(monkeypatch, "What is on not-a-date?") == 0
    assert capsys.readouterr().out == "That is not a valid date.\n"
    (log,) = created_loggers
    entries = read_log(log.path)
    (tool_result,) = [e for e in entries if e["event"] == "tool_result"]
    assert tool_result["ok"] is False  # the failed step is visible in the log ...
    assert events(entries)[-1] == "final"  # ... but the run still ends with an answer
