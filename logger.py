"""
Structured logging and CloudWatch metrics for Alexa-LLM.

Two outputs per invocation:
  1. log_invocation() — plain structured JSON for CloudWatch Logs Insights queries
  2. emit_metrics()   — CloudWatch Embedded Metrics Format (EMF) line that
                        CloudWatch automatically extracts as custom metrics
                        (no boto3 API call, no added latency)

Metrics published under the "AlexaLLM" namespace:
  - Invocations  (Count)       — one per request, dimension: Intent
  - Errors       (Count)       — 1 on failure, 0 on success
  - Latency      (Milliseconds) — total Lambda handler duration
"""

import json
import time

_NAMESPACE = "AlexaLLM"


def log_invocation(intent, user_id, latency_ms, error=None):
    """Emit a structured JSON log line for every Lambda invocation."""
    entry = {
        "event": "invocation",
        "intent": intent,
        "userId": user_id,
        "latency_ms": latency_ms,
        "error": str(error) if error else None,
    }
    print(json.dumps(entry))


def emit_metrics(intent, latency_ms, error=None):
    """Emit a CloudWatch Embedded Metrics Format line.

    CloudWatch Logs automatically parses lines containing the _aws key and
    publishes the named metrics without any SDK call.
    """
    entry = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": _NAMESPACE,
                    "Dimensions": [["Intent"]],
                    "Metrics": [
                        {"Name": "Invocations", "Unit": "Count"},
                        {"Name": "Errors", "Unit": "Count"},
                        {"Name": "Latency", "Unit": "Milliseconds"},
                    ],
                }
            ],
        },
        "Intent": intent,
        "Invocations": 1,
        "Errors": 1 if error else 0,
        "Latency": latency_ms,
    }
    print(json.dumps(entry))
