"""Shared fixtures. They make sure no test touches the network or really sleeps."""

import pytest
import requests

from agent import agent as agent_module
from agent.logger import RunLogger
from tests.helpers import CFG, ScriptedLLM


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if a test accidentally sends a real HTTP request.

    Tests that need HTTP replace requests.get again with their own fake.
    """

    def blocked(*args, **kwargs):
        raise AssertionError("Real HTTP request in a test")

    monkeypatch.setattr(requests, "get", blocked)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Replace time.sleep so retries do not slow the tests down.

    The fixture returns the list of requested waiting times.
    """
    waited = []
    monkeypatch.setattr(agent_module.time, "sleep", waited.append)
    return waited


@pytest.fixture
def start_run(monkeypatch, tmp_path):
    """Prepare the agent loop with a scripted model.

    Usage: llm, log = start_run(reply(...), reply(...), calendar=FakeCalendar())
    Then call agent_module.run("task", CFG, log).
    """
    loggers = []

    def _start(*script, calendar=None, day_range=("2026-10-26", "2026-11-08")):
        llm = ScriptedLLM(*script)

        def make_client(**kwargs):
            llm.client_kwargs = kwargs
            return llm

        monkeypatch.setattr(agent_module, "OpenAI", make_client)
        monkeypatch.setattr(
            agent_module, "get_calendar_range", lambda base_url: day_range
        )
        if calendar is not None:
            monkeypatch.setattr(agent_module, "get_calendar_day", calendar)
        log = RunLogger(CFG["MODEL"], log_dir=str(tmp_path / "logs"))
        loggers.append(log)
        return llm, log

    yield _start

    for log in loggers:
        log.close()
