"""Reads the configuration from the environment and the .env file."""

import os

from dotenv import load_dotenv

# these must be set, either in real environment or in .env file
REQUIRED = ["OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL"]


def load_config() -> dict:
    """Loads variables from .env into os.environ.
    Variables that are already set in real environment are overwritten.
    """

    load_dotenv()
    missing = [k for k in REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env variable: {', '.join(missing)}")
    cfg = {k: os.environ[k] for k in REQUIRED}
    # remove trailing slash so URLs later do not produce "//"
    cfg["DATA_BASE_URL"] = cfg["DATA_BASE_URL"].rstrip("/")
    return cfg
