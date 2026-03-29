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
    _get_table().put_item(Item={
        "userId": user_id,
        "history": history,
    })
