"""Project-wide logger factory.

One file handler per named logger, written to ``Data/logs/rag_app.log``. We
disable propagation so module loggers don't also dump through the root logger
when something upstream (pytest, streamlit, uvicorn) installs its own handler.
"""

from __future__ import annotations

import logging

import config


def get_logger(name: str = "rag") -> logging.Logger:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    # Re-entrant safe: if someone has already configured this named logger,
    # hand it back as-is rather than stacking a second identical handler.
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(config.APP_LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
