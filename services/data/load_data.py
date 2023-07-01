import sys
sys.path.append('..')

import os
import urllib

import boto3
import weaviate as wv
from utils import content


lambda_data_dir = '/tmp'

s3_client = boto3.client('s3')

auth_config = wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

wv_client = wv.Client(
    url=os.environ['WEAVIATE_URL'],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
    },
    auth_client_secret=auth_config
)


def handler(event, context):
    bucket = urllib.parse.unquote_plus(event['Records'][0]['s3']['bucket']['name'], encoding='utf-8')
    doc_name = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    print(doc_name)

    data_class = doc_name.split('/')[-2]
    report_path = lambda_data_dir + '/{}'.format(doc_name.split('/')[-1])
    s3_client.download_file(bucket, doc_name, report_path)
    
    if doc_name.endswith('.pdf') or doc_name.endswith('.json'):
        content.load_content(
            report_path,
            data_class=data_class,
            chunk_size=100,
            wv_client=wv_client
        )
    else:
        print("NOT PROCESSING - Doesn't end in .pdf or .json")

