import sys
sys.path.append('..')

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

import argparse
from datetime import datetime
import time
import json
import boto3
import pandas as pd
from utils.bubble import (
    create_bubble_object,
    get_bubble_object,
    update_bubble_object,
    get_bubble_doc,
    delete_bubble_object,
    upload_bubble_file
)
from utils.cloud_funcs import cloud_ecs
from utils.general import SQS
from utils.response_lib import *
from utils.workflow_utils import make_batch_files


STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

#if os.getenv('IS_OFFLINE'):
#   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
#   LAMBDA_DATA_DIR = '.'
#else:
#   lambda_client = boto3.client('lambda')
#   LAMBDA_DATA_DIR = '/tmp'
LAMBDA_DATA_DIR = '.'




def main(task_args):
    print('task_args:')
    print(task_args)
    print('')

    email = task_args['user_email']
    ec_lib_name = task_args['ec_lib_name']
    domain = task_args['customer_domain']
    top_n_ser = task_args['top_n_ser']

    # Fetch batch input file from bubble
    batch_url = task_args['batch_input_url']
    batch_input_file_name = batch_url.split('/')[-1]
    local_batch_path = f'{LAMBDA_DATA_DIR}/{batch_input_file_name}'

    if 'app.foxscript.ai' in batch_url:
        get_bubble_doc(batch_url, local_batch_path)
        print("Retrieved batch doc from bubble")
    else:
        local_batch_path = batch_url
        print("Using local batch file")

    # Get Topics
    topics_df = pd.read_csv(local_batch_path)
    print(f"Topics Shape: {topics_df.shape}")
    topics = [t.split(' - ')[0] for t in topics_df.Keyword]

    # load batch list
    #if local_batch_path.endswith('.csv'):
    #    print('Preparing CSV Batch Input')
    #    batch_list = get_csv_lines(content=None, path=local_batch_path, delimiter=',', return_as_json=True)
    #    batch_list = [item for item in batch_list if item != '']
    #else:
    #    print('PROBLEM: Batch Input Not CSV')

    
    # Process the Batch CSV
    sqs = 'ecs{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    counter = 0
    serper_api = False
    for idx, topic in enumerate(topics):
        if idx%10 == 0:
            print(f"Item #{idx}: {topic}")

        # do distributed ECS for each topic
        if topic:
            cloud_ecs(topic, ec_lib_name, email, domain, top_n_ser, serper_api, sqs=sqs, invocation_type='Event') 
            time.sleep(0.1)
            counter += 1
            serper_api = not serper_api
        else:
            pass

    # wait for and collect search results from SQS
    all_ecs = queue.collect(counter, max_wait=600)
    print(f"all_ecs length: {len(all_ecs)}")

    ecs_df = pd.DataFrame(all_ecs)
    print(f'ECS DF SHAPE: {ecs_df.shape}')

    domain_name = domain.split('.')[0]
    ecs_file_name = f'{domain_name}_ecs.csv'
    local_ecs_path = f'{LAMBDA_DATA_DIR}/{ecs_file_name}'
    print(local_ecs_path)
    ecs_df.to_csv(local_ecs_path, index=False)
        
    # send result to Bubble Document
    #body = {}
    #res = create_bubble_object('document', body)
    #new_doc_id = res.json()['id']

    # add new doc to project
    #res = get_bubble_object('project', project_id)
    #try:
    #    project_docs = res.json()['response']['documents']
    #except:
    #    project_docs = []

    #_ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]})




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_args', default=None, type=str)
    args, _ = parser.parse_known_args()
    print(args.task_args)

    task_args = json.loads(args.task_args)

    start = datetime.now()
    print(start)
    main(task_args)
    print(datetime.now() - start)

  