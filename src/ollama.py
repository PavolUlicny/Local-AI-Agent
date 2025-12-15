"""Helpers for detecting and starting the Ollama runtime.

This module exposes small, testable functions for:
- checking whether the `ollama` CLI is installed,
- probing the local Ollama HTTP API for readiness,
- starting `ollama serve` detached and waiting until the API is responding,
- a convenience `ensure_available()` that returns a boolean status rather
  than exiting the process.

These helpers are intentionally side-effect-light: they log via the
module logger and return boolean statuses so callers (for example `src.main`)
can decide whether to exit or continue in degraded mode.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_PORT = 11434
DEFAULT_HOST = "127.0.0.1"
DEFAULT_LOG_PATH = "~/.local/share/ollama/installer_ollama.log"

logger = logging.getLogger(__name__)


def is_installed() -> bool:
    """Return True if the `ollama` CLI is available on PATH."""
    return shutil.which("ollama") is not None


def is_ready(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    """Return True if the Ollama HTTP API responds at `/api/tags`.

    Uses a short timeout and is intended for polling readiness.
    """
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except urllib.error.URLError:
        return False
    except socket.timeout:
        return False


def start_detached(log_path: str | None = None) -> Optional[subprocess.Popen]:
    """Start `ollama serve` detached and return the Popen object on success.

    If `log_path` is provided (or the default), stdout/stderr are redirected
    to that file; otherwise they are suppressed. On failure the function
    returns `None` and logs an error.
    """
    if not is_installed():
        logger.error("Ollama CLI not found on PATH")
        return None

    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    log = None
    if log_path is None:
        log_path = DEFAULT_LOG_PATH
    try:
        log = open(os.path.expanduser(log_path), "a+")
    except Exception:
        logger.debug("Unable to open Ollama log file '%s', proceeding without log file", log_path)
        log = None

    try:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=log or subprocess.DEVNULL,
            stderr=log or subprocess.STDOUT,
            start_new_session=True,
            creationflags=creationflags,
        )
        pid = getattr(proc, "pid", None)
        if pid:
            logger.debug("Started ollama serve (pid %s)", pid)
        return proc
    except FileNotFoundError:
        logger.error("Failed to start Ollama: 'ollama' not found at execution time")
        if log:
            log.close()
        return None
    except Exception as e:
        logger.error("Failed to start Ollama: %s", e)
        if log:
            log.close()
        return None


def wait_for_ready(
    wait_seconds: int = 30, poll_interval: float = 1.0, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> bool:
    """Poll the Ollama HTTP API until it responds or the timeout expires.

    Returns True when the API responds, False otherwise.
    """
    deadline = time.time() + float(wait_seconds)
    while time.time() < deadline:
        if is_ready(host=host, port=port):
            logger.debug("Ollama server is up and responding")
            return True
        time.sleep(poll_interval)
    logger.error("Ollama did not become ready within %s seconds", wait_seconds)
    return False


def ensure_available(wait_seconds: int = 30, log_path: str | None = None, poll_interval: float = 1.0) -> bool:
    """Ensure Ollama is installed and responding.

    Returns True when the Ollama HTTP API is responding (either already
    running or successfully started). Returns False when Ollama is not
    installed or could not be started/respond in time.
    """
    if not is_installed():
        logger.error("Ollama CLI not found on PATH. Please install Ollama and re-run.")
        return False

    if is_ready():
        return True

    # try to start
    proc = start_detached(log_path=log_path)
    if proc is None:
        return False

    ok = wait_for_ready(wait_seconds=wait_seconds, poll_interval=poll_interval)
    if not ok:
        return False
    return True


def check_and_start_ollama(
    wait_seconds: int = 30,
    log_path: str | None = None,
    poll_interval: float = 1.0,
    exit_on_failure: bool = False,
) -> bool:
    """Run the full Ollama availability workflow and emit concise logs.

    This convenience wrapper performs the same actions as `ensure_available`
    but provides higher-level `INFO`/`ERROR` logging so callers can invoke a
    single function to perform the check/start/wait sequence.

    Returns True when Ollama is available (already running or started and
    responding). Returns False when Ollama is missing or could not be started
    within the timeout. If `exit_on_failure` is True the function will call
    `sys.exit(1)` on failure; otherwise it returns False and leaves the
    decision to the caller.
    """
    import sys as _sys

    logger.info("Verifying Ollama availability...")

    ok = ensure_available(wait_seconds=wait_seconds, log_path=log_path, poll_interval=poll_interval)
    if ok:
        logger.info("Ollama is available and responding.")
        return True

    logger.error("Ollama is unavailable after attempting to start it.")
    if exit_on_failure:
        _sys.exit(1)
    return False


# Public API: only expose the high-level wrapper by default
__all__ = ["check_and_start_ollama"]
