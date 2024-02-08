import os
import json
import boto3

STAGE = os.getenv('STAGE')
lambda_client = boto3.client('lambda')


def cloud_google_search(q:str, n:int=None, sqs:str=None):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-data-{STAGE}-google_search',
        InvocationType='Event' if sqs else 'RequestResponse',
        Payload=json.dumps({"body": {
            'q': q,
            'n': n,
            'sqs': sqs
        }})
    )

    return res

def cloud_scrape(url, sqs=None, invocation_type='Event', chunk_overlap=10, return_raw=False):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-data-{STAGE}-scraper',
        InvocationType=invocation_type,
        Payload=json.dumps({"body": {
            'url': url,
            'sqs': sqs,
            'chunk_overlap': chunk_overlap,
            'return_raw': return_raw
        }})
    )

    return res


def cloud_research(url, sqs=None, query=None, invocation_type='Event', chunk_overlap=10):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-data-{STAGE}-researcher',
        InvocationType=invocation_type,
        Payload=json.dumps({"body": {
            'url': url,
            'sqs': sqs,
            'query': query,
            'chunk_overlap': chunk_overlap
        }})
    )

    return res