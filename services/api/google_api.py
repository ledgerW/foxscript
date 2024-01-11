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
from utils.google import get_creds, create_drive_folder, search_drive_folders


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'



# Lambda Handler
def get_drive_folders(event, context):
    print(event)

    try:
        goog_token = event['body']['goog_token']
        goog_refresh_token = event['body']['goog_refresh_token']
        parent_id = event['body']['parent_id']
    except:
        goog_token = json.loads(event['body'])['goog_token']
        goog_refresh_token = json.loads(event['body'])['goog_refresh_token']
        parent_id = json.loads(event['body'])['parent_id']
 
    creds = get_creds(goog_token, goog_refresh_token)

    folders = search_drive_folders(parent_id=parent_id, creds=creds)

    return success({'folders': folders})


def make_drive_folder(event, context):
    print(event)

    try:
        goog_token = event['body']['goog_token']
        goog_refresh_token = event['body']['goog_refresh_token']
        name = event['body']['name']
        parent_id = event['body']['parent_id']
    except:
        goog_token = json.loads(event['body'])['goog_token']
        goog_refresh_token = json.loads(event['body'])['goog_refresh_token']
        name = json.loads(event['body'])['name']
        parent_id = json.loads(event['body'])['parent_id']
 
    creds = get_creds(goog_token, goog_refresh_token)

    folder_id = create_drive_folder(name, parents=parent_id, creds=creds)

    return success({'folder_id': folder_id})