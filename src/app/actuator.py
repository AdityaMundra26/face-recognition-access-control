"""The real-world action a granted access check triggers.

Default is log-only, so a fresh checkout or a misconfigured deployment
never silently drives real hardware. Set ACCESS_WEBHOOK_URL to POST a JSON
payload to an HTTP-triggered relay/door-controller instead (e.g. a Shelly
relay, a Home Assistant webhook, or any similar HTTP-triggered door strike)
-- this project has no physical hardware to integrate directly, so a
webhook is the most it can responsibly do out of the box.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

WEBHOOK_ENV_VAR = "ACCESS_WEBHOOK_URL"
WEBHOOK_TIMEOUT_SECONDS = 5


def grant_access(name: str) -> None:
    """Perform whatever real-world action "granting access" means here.

    Never raises: a webhook failure is logged, not surfaced to the caller,
    since the recognition decision has already been made and the UI has
    already told the user they were granted access.
    """
    webhook_url = os.environ.get(WEBHOOK_ENV_VAR)
    if not webhook_url:
        logger.info("ACCESS GRANTED for %s (no %s configured; log-only)", name, WEBHOOK_ENV_VAR)
        return

    try:
        response = requests.post(
            webhook_url, json={"event": "access_granted", "name": name}, timeout=WEBHOOK_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.exception("Failed to notify access webhook for %s", name)
