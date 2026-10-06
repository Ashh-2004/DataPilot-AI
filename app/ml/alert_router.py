"""Webhook routing for anomaly notifications."""

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


class AlertRouter:
    """Send anomaly notifications to the configured n8n webhook."""

    def __init__(self) -> None:
        """Load the webhook URL and timeout from the environment."""
        load_dotenv()
        self.webhook_url = os.getenv("N8N_WEBHOOK_URL", "").strip()
        self.timeout = float(os.getenv("N8N_WEBHOOK_TIMEOUT_SECONDS", "5"))

    def send(self, anomaly_result: dict[str, Any], query: str) -> None:
        """Post an anomaly payload and log failures without raising them."""
        if not self.webhook_url:
            return
        columns = anomaly_result.get("anomaly_columns", [])
        explanation = (
            f"Column {', '.join(str(column) for column in columns)} drove the anomaly."
            if columns
            else "No contributing columns were identified."
        )
        payload = {
            "query": query,
            "anomaly_count": int(anomaly_result.get("anomaly_count", 0)),
            "flagged_rows": anomaly_result.get("anomaly_rows", []),
            "explanation": explanation,
        }
        try:
            response = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            LOGGER.warning("Anomaly webhook failed for %s", self.webhook_url, exc_info=True)
