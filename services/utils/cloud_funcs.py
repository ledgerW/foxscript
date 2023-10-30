import os
import json
import boto3


STAGE = os.getenv('STAGE')

if os.getenv('IS_OFFLINE'):
   #boto3.setup_default_session(profile_name='ledger')
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
else:
   lambda_client = boto3.client('lambda')


def cloud_scrape(url, sqs=None, query=None):
  _ = lambda_client.invoke(
    FunctionName=f'foxscript-data-{STAGE}-scraper',
    InvocationType='Event',
    Payload=json.dumps({"body": {
        'url': url,
        'sqs': sqs,
        'query': query
      }})
  )


def cloud_research(url, sqs=None, query=None):
  _ = lambda_client.invoke(
    FunctionName=f'foxscript-data-{STAGE}-researcher',
    InvocationType='Event',
    Payload=json.dumps({"body": {
        'url': url,
        'sqs': sqs,
        'query': query
      }})
  )