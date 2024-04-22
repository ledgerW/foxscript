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
SERVICE = os.getenv('SERVICE')

if os.getenv('IS_OFFLINE'):
   LAMBDA_DATA_DIR = '.'
else:
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
        taskDefinition=f'run_task-{STAGE}',
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
                'name': f'run_task-{STAGE}',
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
    if task_name == 'run_task_ecs':
        from run_task_ecs import main
    
    main(task_args)


def handler(event, context):
    print(event)
    try:
        input = json.loads(event['body'])
    except:
        input = event['body']

    task_name = input['task']
    task_args = input['task_args']

    print('task_api task_args:')
    print(task_args)
    print('')

    task_script = f'run_task_{task_name}'

    if os.getenv('IS_OFFLINE', 'false') == 'true':
        print('RUNNING LOCAL')
        run_local(task_script, task_args=task_args)
    else:
        print('RUNNING CLOUD')
        run_cloud(task_script, task_args=task_args)

    return success({'success': True})