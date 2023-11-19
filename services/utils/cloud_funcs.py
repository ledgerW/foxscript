import os
import json
import boto3

STAGE = os.getenv('STAGE')


if os.getenv('IS_OFFLINE'):
    import sys
    sys.path.append('..')

    from utils.general import SQS
    
    lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))


    def cloud_scrape(url, sqs=None):
        event = {'body': {
          'url': url,
          'sqs': sqs
        }}

        chunks = '<SPLIT>'.join([f'Some dummy text for {url}', f'and some more dummy text for {url}'])
        res = {'url': url, 'chunks': chunks}

        queue = SQS(sqs)
        queue.send(res)

        return res
    

    def cloud_research(url, sqs=None, query=None):
        event = {'body': {
          'url': url,
          'sqs': sqs,
          'query': query
        }}

        result = "Some dummy text."
        res = {
            'output': result,
            'input_word_cnt': len(query.split(' ')),
            'output_word_cnt': len(result.split(' '))
        }
       
        queue = SQS(sqs)
        queue.send(res)

        return res
    
else:
    lambda_client = boto3.client('lambda')

    def cloud_scrape(url, sqs=None):
      _ = lambda_client.invoke(
        FunctionName=f'foxscript-data-{STAGE}-scraper',
        InvocationType='Event',
        Payload=json.dumps({"body": {
            'url': url,
            'sqs': sqs
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