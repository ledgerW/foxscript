import sys
sys.path.append('..')

import os
import urllib
import json

import boto3
from utils.content import load_content
from utils.weaviate_utils import wv_client, to_json_doc


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
  LAMBDA_DATA_DIR = '.'
else:
  LAMBDA_DATA_DIR = '/tmp'

s3_client = boto3.client('s3')


def handler(event, context):
    print(event)
    # S3 Trigger
    if 'Records' in event:
        bucket = urllib.parse.unquote_plus(event['Records'][0]['s3']['bucket']['name'], encoding='utf-8')
        doc_name = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')

        data_class = doc_name.split('/')[-2]
        local_content_path = LAMBDA_DATA_DIR + '/{}'.format(doc_name.split('/')[-1])
        s3_client.download_file(bucket, doc_name, local_content_path)
    else:
        # HTTP Trigger
        try:
            bucket = event['body']['bucket']
            data_class = event['body']['cls_name']
            content = event['body']['content']
            body = event['body']
        except:
            bucket = json.loads(event['body'])['bucket']
            data_class = json.loads(event['body'])['cls_name']
            content = json.loads(event['body'])['content']
            body = json.loads(event['body'])

        if 'url' in body:
            url = body['url']
        else:
            url = ""

        local_content_path, upload_suffix = to_json_doc(data_class, content, url=url)
    
    if local_content_path.endswith('.pdf') or local_content_path.endswith('.json'):
        load_content(
            local_content_path,
            data_class=data_class,
            chunk_size=100,
            wv_client=wv_client
        )
        print(f"Loaded {local_content_path}")
    else:
        print("NOT PROCESSING - Doesn't end in .pdf or .json")

