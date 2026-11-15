"""Tests for agent/config.py."""

import pytest

from agent import config

REQUIRED = ["OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # load_dotenv would read a real .env file of the developer and could hide
    # missing variables, so it is replaced and the variables are cleared.
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: None)
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)


def set_env(monkeypatch, **overrides):
    values = {
        "OPENAI_BASE_URL": "https://llm.test/v1",
        "MODEL": "test-model",
        "OPENAI_API_KEY": "test-key",
        "DATA_BASE_URL": "https://data.test/v1",
    }
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_returns_exactly_the_four_variables(monkeypatch):
    set_env(monkeypatch)
    cfg = config.load_config()
    assert cfg == {
        "OPENAI_BASE_URL": "https://llm.test/v1",
        "MODEL": "test-model",
        "OPENAI_API_KEY": "test-key",
        "DATA_BASE_URL": "https://data.test/v1",
    }


@pytest.mark.parametrize(
    "raw", ["https://data.test/v1", "https://data.test/v1/", "https://data.test/v1//"]
)
def test_trailing_slashes_of_the_data_url_are_removed(monkeypatch, raw):
    set_env(monkeypatch, DATA_BASE_URL=raw)
    assert config.load_config()["DATA_BASE_URL"] == "https://data.test/v1"


def test_only_the_data_url_is_normalised(monkeypatch):
    set_env(monkeypatch, OPENAI_BASE_URL="https://llm.test/v1/")
    assert config.load_config()["OPENAI_BASE_URL"] == "https://llm.test/v1/"


@pytest.mark.parametrize("missing", REQUIRED)
def test_a_missing_variable_is_reported_by_name(monkeypatch, missing):
    set_env(monkeypatch)
    monkeypatch.delenv(missing)
    with pytest.raises(RuntimeError, match=missing):
        config.load_config()


def test_an_empty_variable_counts_as_missing(monkeypatch):
    set_env(monkeypatch, MODEL="")
    with pytest.raises(RuntimeError, match="MODEL"):
        config.load_config()


def test_all_missing_variables_are_listed_together():
    with pytest.raises(RuntimeError) as excinfo:
        config.load_config()
    for name in REQUIRED:
        assert name in str(excinfo.value)


def test_the_env_file_is_loaded_at_start(monkeypatch):
    set_env(monkeypatch)
    calls = []
    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: calls.append(1))
    config.load_config()
    assert calls == [1]