import logging
import sys


def setup_logger(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)
    if not root.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(h)

    # Keep returning the project logger for backward compatibility,
    # but let it propagate so module loggers are visible too.
    log = logging.getLogger("overconflens")
    log.setLevel(lvl)
    log.propagate = True
    return log
