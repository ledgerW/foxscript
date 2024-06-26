import sys
sys.path.append('..')

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import os
import time
import json
import uuid
import boto3

from utils.response_lib import *
from utils.bubble import update_bubble_object


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')
SERVICE = os.getenv('SERVICE')
ECS_SQS_URL = os.getenv('ECS_SQS_URL')
ECS_SQS_NAME = os.getenv('ECS_SQS_NAME')
ECS_GROUP_MESSAGE_ID = os.getenv('ECS_GROUP_MESSAGE_ID')
SAMPLE_ECS_SQS_URL = os.getenv('SAMPLE_ECS_SQS_URL')
SAMPLE_ECS_SQS_NAME = os.getenv('SAMPLE_ECS_SQS_NAME')
SAMPLE_ECS_GROUP_MESSAGE_ID = os.getenv('SAMPLE_ECS_GROUP_MESSAGE_ID')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')



def pushlish_ecs_message(message, sample_job=False):
    """
    message = {
        'task': 'ecs',
        'task_args': {
            "ecs_job_id": "1716399273224x459847603399916162",
            "user_email": "ledger.west@gmail.com",
            "top_n_ser": 2,
            "ecs_concurrency": 50
        }
    }
    """
    sqs = boto3.client('sqs')

    URL = SAMPLE_ECS_SQS_URL if sample_job else ECS_SQS_URL
    GROUP_MESSAGE_ID = SAMPLE_ECS_GROUP_MESSAGE_ID if sample_job else ECS_GROUP_MESSAGE_ID
    
    res = sqs.send_message(
        QueueUrl=URL,
        MessageBody=json.dumps(message),
        MessageDeduplicationId=str(uuid.uuid1()),
        MessageGroupId=GROUP_MESSAGE_ID,
        MessageAttributes={}
    )

    return res


def get_ecs_message(sample_job=False) -> dict:
    sqs = boto3.client('sqs')

    URL = SAMPLE_ECS_SQS_URL if sample_job else ECS_SQS_URL
    
    res = sqs.receive_message(
        QueueUrl=URL,
        MessageAttributeNames=['All']
    )

    try:
        message = json.loads(res['Messages'][0]['Body'])

        res = sqs.delete_message(
            QueueUrl=URL,
            ReceiptHandle=res['Messages'][0]['ReceiptHandle']
        )

        return message
    except:
        return None
    

def get_ecs_message_count(sample_job=False) -> int:
    sqs = boto3.client('sqs')

    URL = SAMPLE_ECS_SQS_URL if sample_job else ECS_SQS_URL
    
    res = sqs.get_queue_attributes(
        QueueUrl=URL,
        AttributeNames=['All']
    )

    return int(res['Attributes']['ApproximateNumberOfMessages'])


def get_ecs_running_jobs_count() -> int:
    ecs_client = boto3.client('ecs')

    # get cluster
    res = ecs_client.list_clusters()
    cluster = [cl for cl in res['clusterArns'] if SERVICE in cl and STAGE in cl][0]
    print(cluster)

    res = ecs_client.list_tasks(
        cluster=cluster,
        desiredStatus='RUNNING'
    )
    
    return len(res['taskArns'])


def check_and_trigger_ecs_job(max_jobs=1, sample_job=False):
    ecs_job_message = None
    ecs_message_count = get_ecs_message_count()
    if ecs_message_count > 0:
        # Is there capacity to run another job (limited by scraper API)?
        ecs_jobs_count = get_ecs_running_jobs_count()
        if ecs_jobs_count < max_jobs:
            # Get next ECS job in queue and run it
            ecs_job_message = get_ecs_message()

            if ecs_job_message:
                # Send it to task_api to run in Fargate
                _ = lambda_client.invoke(
                    FunctionName=f'foxscript-task-{STAGE}-task',
                    InvocationType='RequestResponse',
                    Payload=json.dumps({"body": ecs_job_message})
                )

                job_body = {
                    'is_running': True
                }
                ecs_job_id = ecs_job_message['task_args']['ecs_job_id']
                res = update_bubble_object('ecs-job', ecs_job_id, job_body)

    return ecs_job_message



def publish(event, context):
    print(event)
    try:
        message = json.loads(event['body'])
    except:
        message = event['body']

    res = pushlish_ecs_message(message)

    message_count = get_ecs_message_count()

    return success({'message_count': message_count})


def poll(event, context):
    """
    body = {
        max_jobs: 1,
        sample_job: True
    }
    """
    print(event)
    try:
        try:
            body = json.loads(event['body'])
        except:
            body = event['body']

        max_jobs = body['max_jobs']
        sample_job = body['sample_job']
    except:
        max_jobs = 1
        sample_job = False

    # Check every 10 seconds
    # Are there any jobs waiting to be executed?
    for _ in range(6):
        ecs_job_message = check_and_trigger_ecs_job(max_jobs, sample_job)
        time.sleep(10)

    return success({'success': True})