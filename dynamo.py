import boto3
import os
import time

_table = None

# Maximum number of conversation turns (user + assistant = 1 turn = 2 messages)
# kept in history. Oldest turns are pruned when this limit is exceeded.
MAX_HISTORY_TURNS = int(os.environ.get("MAX_HISTORY_TURNS", "20"))

# TTL for DynamoDB items: 7 days of inactivity before automatic expiry.
# Requires TTL to be enabled on the table pointing at the 'ttl' attribute.
_TTL_SECONDS = 7 * 24 * 60 * 60


def _ttl_timestamp():
    return int(time.time()) + _TTL_SECONDS


def _get_table():
    global _table
    if _table is None:
        table_name = os.environ.get("DYNAMODB_TABLE", "AlexaConversationHistory")
        _table = boto3.resource("dynamodb").Table(table_name)
    return _table


def get_history(user_id):
    response = _get_table().get_item(Key={"userId": user_id})
    return response.get("Item", {}).get("history", [])


def prune_history(history, max_turns=None):
    """Return history trimmed to the most recent max_turns turns.

    Each turn is one user message + one assistant message (2 list entries).
    If history is already within the limit it is returned unchanged.
    """
    if max_turns is None:
        max_turns = MAX_HISTORY_TURNS
    max_messages = max_turns * 2
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


def save_history(user_id, history):
    pruned = prune_history(history)
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="SET history = :h, #ttl = :t",
        ExpressionAttributeNames={"#ttl": "ttl"},
        ExpressionAttributeValues={":h": pruned, ":t": _ttl_timestamp()},
    )


def get_user_facts(user_id):
    response = _get_table().get_item(Key={"userId": user_id})
    return response.get("Item", {}).get("userFacts", {})


def merge_user_facts(user_id, new_facts):
    existing = get_user_facts(user_id)
    existing.update(new_facts)
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="SET userFacts = :f",
        ExpressionAttributeValues={":f": existing},
    )


def clear_user_facts(user_id):
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="REMOVE userFacts",
    )


def clear_user_fact(user_id, key):
    facts = get_user_facts(user_id)
    if key in facts:
        del facts[key]
        _get_table().update_item(
            Key={"userId": user_id},
            UpdateExpression="SET userFacts = :f",
            ExpressionAttributeValues={":f": facts},
        )


def get_pending_chunks(user_id):
    response = _get_table().get_item(Key={"userId": user_id})
    return response.get("Item", {}).get("pendingChunks", [])


def save_pending_chunks(user_id, chunks):
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="SET pendingChunks = :c",
        ExpressionAttributeValues={":c": chunks},
    )


def clear_pending_chunks(user_id):
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="REMOVE pendingChunks",
    )
