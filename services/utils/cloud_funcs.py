import os
import json
import boto3

STAGE = os.getenv('STAGE')


if os.getenv('IS_OFFLINE'):
    import sys
    sys.path.append('..')

    from utils.general import SQS
    

    def cloud_scrape(url, sqs=None, invocation_type='Event', chunk_overlap=10):
        event = {'body': {
          'url': url,
          'sqs': sqs,
          'chunk_overlap': chunk_overlap
        }}

        if invocation_type == 'Event':
            chunks = '<SPLIT>'.join([f'Some dummy text for {url}', f'and some more dummy text for {url}'])
            res = {'url': url, 'chunks': chunks}

            queue = SQS(sqs)
            queue.send(res)
        else:
            lambda_client = boto3.client('lambda')
            
            res = lambda_client.invoke(
                FunctionName=f'foxscript-data-{STAGE}-scraper',
                InvocationType=invocation_type,
                Payload=json.dumps({"body": {
                    'url': url,
                    'sqs': sqs,
                    'chunk_overlap': chunk_overlap
                }})
            )
            

        return res
    

    def cloud_research(url, sqs=None, query=None, invocation_type='Event', chunk_overlap=10):
        event = {'body': {
          'url': url,
          'sqs': sqs,
          'query': query,
          'chunk_overlap': chunk_overlap
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

    def cloud_scrape(url, sqs=None, invocation_type='Event', chunk_overlap=10):
        res = lambda_client.invoke(
            FunctionName=f'foxscript-data-{STAGE}-scraper',
            InvocationType=invocation_type,
            Payload=json.dumps({"body": {
                'url': url,
                'sqs': sqs,
                'chunk_overlap': chunk_overlap
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