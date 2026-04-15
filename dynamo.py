import boto3
import os

_table = None


def _get_table():
    global _table
    if _table is None:
        table_name = os.environ.get("DYNAMODB_TABLE", "AlexaConversationHistory")
        _table = boto3.resource("dynamodb").Table(table_name)
    return _table


def get_history(user_id):
    response = _get_table().get_item(Key={"userId": user_id})
    return response.get("Item", {}).get("history", [])


def save_history(user_id, history):
    _get_table().update_item(
        Key={"userId": user_id},
        UpdateExpression="SET history = :h",
        ExpressionAttributeValues={":h": history},
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
