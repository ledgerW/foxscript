import sys
sys.path.append('..')

import os
import json
import time
import boto3
import pandas as pd

from utils.bubble import get_bubble_doc, upload_bubble_file
from utils.response_lib import *
from utils.workflow_utils import make_batch_files


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')
SERVICE = os.getenv('SERVICE')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'




def run_cloud(task_name, task_args={}):
    cf_client = boto3.client('cloudformation')
    stackname = f'foxscript-task-{STAGE}'
    response = cf_client.describe_stacks(StackName=stackname)
    outputs = response["Stacks"][0]['Outputs']

    SUBNET1 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet1'][0]
    SUBNET2 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet2'][0]
    SUBNET3 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet3'][0]
    SECURITY_GROUP = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSecurityGroupId'][0]

    # Build task command
    base_cmd = [f"{task_name}.py"]

    task_cmd = ['--task_args'] + [json.dumps(task_args)]

    command = base_cmd + task_cmd

    client = boto3.client('ecs')

    # get cluster
    res = client.list_clusters()
    cluster = [cl for cl in res['clusterArns'] if SERVICE in cl and STAGE in cl][0]

    # run task
    res = client.run_task(
        cluster=cluster, 
        launchType='FARGATE',
        taskDefinition=f'{task_name}-{STAGE}',
        count=1,
        platformVersion='LATEST',
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    SUBNET1,
                    SUBNET2,
                    SUBNET3
                ],
                'securityGroups': [
                    SECURITY_GROUP,
                ],
                'assignPublicIp': 'ENABLED'
            }
        },
        overrides={
        'containerOverrides': [
            {
                'name': f'{task_name}-{STAGE}',
                'command': command,
                'environment': [
                    {
                    'name': 'string',
                    'value': 'string'
                    }
                ]
            }
        ]
        }
    )


def run_local(task_name, task_args={}):
    if task_name == 'run_batch_workflow':
        from run_batch_workflow import main
    
    if task_name == 'run_batch_upload_to_s3':
        from run_batch_upload_to_s3 import main
    
    main(task_args)


def batch_workflow(event, context):
    print(event)
    try:
        task_args = json.loads(event['body'])
    except:
        task_args = event['body']

    print('batch_api task_args:')
    print(type(task_args))
    print(task_args)
    print('')

    # Prep batches according to concurrency
    batch_url = task_args['batch_input_url']
    batch_concurrent_runs = int(task_args['batch_concurrent_runs'])

    # Fetch primary batch input file from bubble
    batch_input_file_name = batch_url.split('/')[-1]
    local_batch_path = f'{LAMBDA_DATA_DIR}/{batch_input_file_name}'
    get_bubble_doc(batch_url, local_batch_path)
    
    batch_df = pd.read_csv(local_batch_path)
    batch_files = make_batch_files(batch_df, concurrent_runs=batch_concurrent_runs, as_csv=True)

    print(f'Starting {batch_concurrent_runs} batch jobs')
    for batch_file in batch_files:
        print(f'Batch_file: {batch_file}')
        bubble_url = upload_bubble_file(batch_file)
        task_args['batch_input_url'] = bubble_url

        if os.getenv('IS_OFFLINE', 'false') == 'true':
            print('RUNNING LOCAL')
            run_local('run_batch_workflow', task_args=task_args)
        else:
            print('RUNNING CLOUD')
            run_cloud('run_batch_workflow', task_args=task_args)

        time.sleep(1)

    return success({'success': True})


def batch_upload_to_s3(event, context):
    print(event)
    try:
        task_args = json.loads(event['body'])
    except:
        task_args = event['body']

    print('batch_api task_args:')
    print(type(task_args))
    print(task_args)
    print('')

    if os.getenv('IS_OFFLINE', 'false') == 'true':
        print('RUNNING LOCAL')
        run_local('run_batch_upload_to_s3', task_args=task_args)
    else:
        print('RUNNING CLOUD')
        run_cloud('run_batch_upload_to_s3', task_args=task_args)

    return success({'success': True})