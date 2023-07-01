import sys
sys.path.append('..')

import os
import json

import boto3
import weaviate as wv
from utils import content
from utils.response_lib import *
from utils.ScraperConfig import Config


STAGE = os.environ['STAGE']
BUCKET = os.environ['BUCKET']

lambda_data_dir = '/tmp'

s3_client = boto3.client('s3')
lambda_client = boto3.client('lambda')

auth_config = wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

wv_client = wv.Client(
    url=os.environ['WEAVIATE_URL'],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
    },
    auth_client_secret=auth_config
)


def master(event, context):
    for source in Config.keys():
        res = lambda_client.invoke(
            FunctionName=f'llmwriter-load-data-{STAGE}-scrape_data_worker',
            InvocationType='Event',
            Payload=json.dumps({'body': {'source': source}})
        )

    return success({'SUCESS': True})


def worker(event, context):
    print(event)
    try:
        source = event['body']['source']
    except:
        source = json.loads(event['body'])['source']

    # Scrape
    res = content.scrape_content(source, wv_client=wv_client)
    res['date'] = res['date'].isoformat()

    if res['content'] != '':
        print('Loading New Content')
        # Save locally as .json
        file_date = res['date'].replace(':', '_')
        local_key = f'{source}/{file_date}.json'
        local_path = lambda_data_dir + f'/{file_date}.json'
        with open(local_path, 'w') as file:
            json.dump(res, file)

        # Upload to Load Data S3 Bucket for Weaviate
        s3_client.upload_file(local_path, BUCKET, local_key)
    else:
        print("No New Content")

    return success({'SUCESS': True})
