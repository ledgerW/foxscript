import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import json
import boto3

from utils.bubble import trigger_bubble_workflow_api
from utils.response_lib import *

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'



# Lambda Handler
def rotate_monthly_workflow_usage(event, context):
    print(event)

    res = trigger_bubble_workflow_api('trigger_rotate_monthly_usage')

    return success(json.loads(res.content))