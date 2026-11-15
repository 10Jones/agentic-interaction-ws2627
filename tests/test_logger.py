"""Tests for agent/logger.py."""

from datetime import datetime

import pytest

from agent.logger import RunLogger
from tests.helpers import read_log

REQUIRED_FIELDS = {"ts", "run_id", "event", "tool", "args", "ok", "model", "duration_ms"}


@pytest.fixture
def make_logger(tmp_path):
    loggers = []

    def _make(model="test-model"):
        log = RunLogger(model, log_dir=str(tmp_path / "logs"))
        loggers.append(log)
        return log

    yield _make

    # Close the files so the temporary directory can be removed on Windows.
    for log in loggers:
        log.close()


def test_creates_the_log_directory_and_file(make_logger, tmp_path):
    log = make_logger()
    assert (tmp_path / "logs").is_dir()
    assert log.path.exists()
    assert log.path.suffix == ".jsonl"


def test_an_existing_log_directory_is_fine(tmp_path, make_logger):
    (tmp_path / "logs").mkdir()
    assert make_logger().path.exists()


def test_an_entry_has_exactly_the_specified_fields(make_logger):
    log = make_logger()
    log.log("user_task")
    (entry,) = read_log(log.path)
    assert set(entry) == REQUIRED_FIELDS


def test_defaults_describe_a_successful_step_without_details(make_logger):
    log = make_logger()
    log.log("user_task")
    (entry,) = read_log(log.path)
    assert entry["ok"] is True
    assert entry["tool"] is None
    assert entry["args"] is None
    assert entry["duration_ms"] is None
    assert entry["model"] == "test-model"
    assert entry["event"] == "user_task"


def test_detail_is_only_written_when_given(make_logger):
    log = make_logger()
    log.log("final", detail="42")
    (entry,) = read_log(log.path)
    assert entry["detail"] == "42"
    assert set(entry) == REQUIRED_FIELDS | {"detail"}


def test_tool_args_duration_and_failure_are_stored(make_logger):
    log = make_logger()
    args = {"date": "2026-10-29", "nested": {"a": [1, 2]}}
    log.log("tool_result", ok=False, tool="get_calendar_day", args=args, duration_ms=185)
    (entry,) = read_log(log.path)
    assert entry["ok"] is False
    assert entry["tool"] == "get_calendar_day"
    assert entry["args"] == args
    assert entry["duration_ms"] == 185


def test_timestamp_is_iso_8601_with_timezone(make_logger):
    log = make_logger()
    log.log("user_task")
    (entry,) = read_log(log.path)
    parsed = datetime.fromisoformat(entry["ts"])
    assert parsed.tzinfo is not None


def test_run_id_is_stable_within_a_run_and_differs_between_runs(make_logger):
    first, second = make_logger(), make_logger()
    first.log("user_task")
    first.log("final")
    second.log("user_task")
    first_ids = {entry["run_id"] for entry in read_log(first.path)}
    second_ids = {entry["run_id"] for entry in read_log(second.path)}
    assert len(first_ids) == 1
    assert first_ids != second_ids
    assert first.path != second.path


def test_entries_keep_their_order(make_logger):
    log = make_logger()
    for event in ["user_task", "tool_call", "tool_result", "final"]:
        log.log(event)
    assert [e["event"] for e in read_log(log.path)] == [
        "user_task",
        "tool_call",
        "tool_result",
        "final",
    ]


def test_non_ascii_text_stays_readable(make_logger):
    log = make_logger()
    text = "Prüfung bei Müller um 10 Uhr ✓ 日本"
    log.log("final", detail=text)
    raw = log.path.read_text(encoding="utf-8")
    assert "Prüfung" in raw
    assert "\\u" not in raw
    assert read_log(log.path)[0]["detail"] == text


def test_a_multiline_detail_stays_on_one_line(make_logger):
    log = make_logger()
    log.log("error", ok=False, detail="line one\nline two")
    assert len(log.path.read_text(encoding="utf-8").splitlines()) == 1
    assert read_log(log.path)[0]["detail"] == "line one\nline two"


def test_entries_are_on_disk_before_the_log_is_closed(make_logger):
    # Needed so that nothing is lost if the run crashes.
    log = make_logger()
    log.log("user_task")
    assert len(read_log(log.path)) == 1


def test_close_can_be_called_twice(make_logger):
    log = make_logger()
    log.log("user_task")
    log.close()
    log.close()
    assert len(read_log(log.path)) == 1