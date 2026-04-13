import logging
import sys


def setup_logger(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("overconflens")
    if log.handlers:
        log.setLevel(getattr(logging, level.upper(), logging.INFO))
        return log
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    log.addHandler(h)
    return log
