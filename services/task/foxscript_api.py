import sys
sys.path.append('..')

import os
import json
import boto3

from utils.response_lib import *


lambda_data_dir = '/tmp'

STAGE = os.environ['STAGE']
SERVICE = os.environ['SERVICE']



def run_cloud(article_url=None, article_path=None):
    cf_client = boto3.client('cloudformation')
    stackname = f'foxscript-task-{STAGE}'
    response = cf_client.describe_stacks(StackName=stackname)
    outputs = response["Stacks"][0]['Outputs']

    SUBNET1 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet1'][0]
    SUBNET2 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet2'][0]
    SUBNET3 = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSubnet3'][0]
    SECURITY_GROUP = [out['OutputValue'] for out in outputs if out['OutputKey']=='AppSecurityGroupId'][0]

    # Build auto task command
    base_cmd = ['hollywood_writer.py']

    if article_url:
        task_cmd = ['--article_url'] + [article_url]

    if article_path:
        task_cmd = ['--article_path'] + [article_path]

    command = base_cmd + task_cmd

    client = boto3.client('ecs')

    # get cluster
    res = client.list_clusters()
    cluster = [cl for cl in res['clusterArns'] if SERVICE in cl and STAGE in cl][0]

    # run task
    res = client.run_task(
        cluster=cluster, 
        launchType='FARGATE',
        taskDefinition=f'foxscript-{STAGE}',
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
                'name': f'foxscript-{STAGE}',
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


def run_local(article_url=None, article_path=None):
    import pathlib
    from utils.scrapers import EWScraper
    from hollywood_writer import main

    if article_url:
        ew_scraper = EWScraper()
        result = ew_scraper.scrape_post(article_url)

        article_name = result['title'][:30].replace(' ', '_')
        article = 'ARTICLE:\n' + result['content']

    if article_path:
        article_name = pathlib.Path(article_path).name.replace('.txt', '')
        with open(article_path) as f:
            article = f.read()
    
    main(article, article_name)



def handler(event, context):
    print(event)
    try:
        article_url = event['article_url']
        article_path = event['article_path']
    except:
        article_url = json.loads(event['article_url'])
        article_path = json.loads(event['article_path'])

    print(article_url)
    print(article_path)

    if os.getenv('IS_OFFLINE', 'false') == 'true':
        print('RUNNING LOCAL')
        run_local(article_url, article_path)
    else:
        print('RUNNING CLOUD')
        run_cloud(article_url, article_path)

    return success({'success': True})