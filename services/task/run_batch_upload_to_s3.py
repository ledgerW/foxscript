import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import argparse
import pathlib
from time import sleep
import boto3

from utils.workflow import prep_input_vals, get_workflow_from_bubble

from utils.bubble import create_bubble_object, get_bubble_object, update_bubble_object, get_bubble_doc, delete_bubble_object
from utils.general import SQS
from utils.response_lib import *

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

lambda_client = boto3.client('lambda')
LAMBDA_DATA_DIR = '/tmp'




def main(task_args):
    print('task_args:')
    print(type(task_args))
    print(task_args)
    print('')

    email = task_args['email']
    name = task_args['name']
    batch_doc_url = task_args['batch_doc_url']
    batch_doc_id = task_args['batch_doc_id']
    library_id = task_args['library_id']

    # Fetch batch input file from bubble
    batch_input_file_name = 'batch_input_list.txt'
    local_batch_path = f'{LAMBDA_DATA_DIR}/{batch_input_file_name}'

    try:
        get_bubble_doc(batch_doc_url, local_batch_path)
        print("Retrieved batch doc from bubble")
    except:
        print("Failed to fetch batch doc from bubble")
        pass

    # load batch list
    batch_input_path = pathlib.Path(local_batch_path)
    with open(batch_input_path, encoding="utf-8") as f:
        batch_list = f.read()

    print('{} items in batch list'.format(len(batch_list.split('\n'))))
    print('batch_list top 10:')
    print(batch_list.split('\n')[:10])

    for doc_url in batch_list.split('\n'):
        print(doc_url)

        out_body = {
            'email': email,
            'name': name,
            'doc_url': doc_url
        }

        _ = lambda_client.invoke(
            FunctionName=f'foxscript-api-{STAGE}-upload_to_s3_cloud',
            InvocationType='Event',
            Payload=json.dumps({"body": out_body})
        )

        # send result to Bubble Library Doc
        body = {
            'name': doc_url,
            'library': library_id,
            'url': doc_url
        }
        res = create_bubble_object('library-doc', body)
        new_doc_id = res.json()['id']

        # add new doc to library
        res = get_bubble_object('library', library_id)
        try:
            library_docs = res.json()['response']['library-docs']
        except:
            library_docs = []

        _ = update_bubble_object('library', library_id, {'library-docs': library_docs+[new_doc_id]})

        sleep(5)
    
    # Delete batch input document
    _ = delete_bubble_object('batch-doc', batch_doc_id)




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_args', default=None, type=str)
  args, _ = parser.parse_known_args()
  print(args.task_args)

  task_args = json.loads(args.task_args)
    
  main(task_args)

  