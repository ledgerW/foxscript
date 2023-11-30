import sys
sys.path.append('..')

import os
import json
import boto3

from utils.response_lib import *


lambda_data_dir = '/tmp'

STAGE = os.environ['STAGE']
SERVICE = os.environ['SERVICE']



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
    from run_batch_workflow import main
    
    main(task_args)



def batch_workflow(event, context):
    print(event)
    task_args = json.loads(event['body'])

    print('batch_api task_args:')
    print(type(task_args))
    print(task_args)
    print('')

    if os.getenv('IS_OFFLINE', 'false') == 'true':
        print('RUNNING LOCAL')
        run_local('run_batch_workflow', task_args=task_args)
    else:
        print('RUNNING CLOUD')
        run_cloud('run_batch_workflow', task_args=task_args)

    return success({'success': True})