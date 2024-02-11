import sys
sys.path.append('..')

try:
  from dotenv import load_dotenv
  load_dotenv()
except:
  pass

import os
import json
import boto3

from utils.response_lib import *


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'



def _create_secret(name: str, secret_value: str) -> dict:
    secrets_client = boto3.client("secretsmanager")
    kwargs = {"Name": name}
    kwargs["SecretString"] = secret_value
    response = secrets_client.create_secret(**kwargs)
    
    return response


def _get_secret(name: str, version: str=None) -> dict:
    secrets_client = boto3.client("secretsmanager")
    kwargs = {'SecretId': name}
    if version is not None:
        kwargs['VersionStage'] = version
    response = secrets_client.get_secret_value(**kwargs)
    
    return response


def _update_secret(name: str, secret_value: str) -> dict:
    secrets_client = boto3.client("secretsmanager")

    kwargs = {'SecretId': name}
    kwargs["SecretString"] = secret_value
    response = secrets_client.update_secret(**kwargs)

    return response


def _delete_secret(name: str, without_recovery: bool=True) -> dict:
    secrets_client = boto3.client("secretsmanager")
    response = secrets_client.delete_secret(
        SecretId=name, ForceDeleteWithoutRecovery=without_recovery)
    
    return response


def get_secret_name(user_id: str, integration: str) -> str:
    return '_'.join([user_id, integration])



# Lambda Handler
def create_secret(event, context):
    print(event)

    try:
        secret_value = event['body']['secret_value']
        integration = event['body']['integration']
        user_id = event['body']['user_id']
    except:
        secret_value = json.loads(event['body'])['secret_value']
        integration = json.loads(event['body'])['integration']
        user_id = json.loads(event['body'])['user_id']

    secret_name = get_secret_name(user_id, integration)

    try:
        res = _create_secret(secret_name, secret_value)
    except:
        res = _update_secret(secret_name, secret_value)

    return success({'Name': res['Name']})


def get_secret(event, context):
    print(event)

    try:
        integration = event['body']['integration']
        user_id = event['body']['user_id']
    except:
        integration = json.loads(event['body'])['integration']
        user_id = json.loads(event['body'])['user_id']

    secret_name = get_secret_name(user_id, integration)
    res = _get_secret(secret_name)
    res = {
        'Name': res['Name'],
        'SecretString': res['SecretString']
    }

    return success(res)


def delete_secret(event, context):
    print(event)

    try:
        integration = event['body']['integration']
        user_id = event['body']['user_id']
    except:
        integration = json.loads(event['body'])['integration']
        user_id = json.loads(event['body'])['user_id']

    secret_name = get_secret_name(user_id, integration)
    res = _delete_secret(secret_name)
    res = {
        'Name': res['Name']
    }

    return success(res)