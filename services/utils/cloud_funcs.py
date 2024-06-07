import os
import json
import boto3

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

STAGE = os.getenv('STAGE')
lambda_client = boto3.client('lambda')


def cloud_get_secret(integration:str, user_id:str):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-get_secret',
        InvocationType='RequestResponse',
        Payload=json.dumps({"body": {
            'integration': integration,
            'user_id': user_id,
        }})
    )

    res_body = json.loads(res['Payload'].read().decode("utf-8"))
    res = json.loads(res_body['body'])

    return res


def cloud_create_secret(secret_vale:str, integration:str, user_id:str):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-get_secret',
        InvocationType='RequestResponse',
        Payload=json.dumps({"body": {
            'secret_vale': secret_vale,
            'integration': integration,
            'user_id': user_id,
        }})
    )

    res_body = json.loads(res['Payload'].read().decode("utf-8"))
    res = json.loads(res_body['body'])

    return res


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
    if return_raw:
        res = lambda_client.invoke(
            FunctionName=f'foxscript-data-{STAGE}-ecs',
            InvocationType=invocation_type,
            Payload=json.dumps({"body": {
                'topic': 'SCRAPE',
                'ec_lib_name': 'SCRAPE',
                'user_email': 'SCRAPE',
                'customer_domain': 'SCRAPE',
                'top_n_ser': 0,
                'urls': [url],
                'special_return': 'scrape',
                'sqs': sqs
            }})
        )

        return res
    else:
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


def cloud_ecs(topic, ec_lib_name, user_email, customer_domain, top_n_ser, urls=[], special_return=None, sqs=None, invocation_type='Event'):
    res = lambda_client.invoke(
        FunctionName=f'foxscript-data-{STAGE}-ecs',
        InvocationType=invocation_type,
        Payload=json.dumps({"body": {
            'topic': topic,
            'ec_lib_name': ec_lib_name,
            'user_email': user_email,
            'customer_domain': customer_domain,
            'top_n_ser': top_n_ser,
            'urls': urls,
            'special_return': special_return,
            'sqs': sqs
        }})
    )

    return res