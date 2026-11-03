"""Entry point: python -m agent "<task>" """

import sys

from .agent import run
from .config import load_config
from .logger import RunLogger


def main() -> int:
    # guarantee windows console utf-8 format to avoid crashes with non-ASCII characters
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    if len(sys.argv) != 2:
        print(
            'Too many arguments, use the following: python -m agent "<task>"',
            file=sys.stderr,
        )
        return 2

    # load config before logger, cause logger need config information.
    # if there is an error with config, there will be no written logs so far
    try:
        cfg = load_config()
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    log = RunLogger(cfg["MODEL"])
    try:
        answer = run(sys.argv[1], cfg, log)
    # if there is any failure (network, API, no answer) it is logged and reported
    except Exception as e:  # noqa: BLE001
        log.log("error", ok=False, detail=str(e))
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
