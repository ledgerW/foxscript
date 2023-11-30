import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import argparse
import pathlib
import boto3

from utils.workflow import prep_input_vals, get_workflow_from_bubble

from utils.bubble import create_bubble_object, get_bubble_object, update_bubble_object, get_bubble_doc, delete_bubble_object
from utils.general import SQS
from utils.response_lib import *

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'




def main(task_args):
    print('task_args:')
    print(type(task_args))
    print(task_args)
    print('')

    workflow_id = task_args['workflow_id']
    email = task_args['email']
    input_vars = task_args['input_vars']
    doc_id = task_args['doc_id']
    batch_url = task_args['batch_input_url']
    batch_doc_id = task_args['batch_doc_id']

    # load and run workflow
    workflow = get_workflow_from_bubble(workflow_id, email=email)

    # get project id for output docs using dummy temp doc id provided in initial call
    res = get_bubble_object('document', doc_id)
    project_id = res.json()['response']['project']

    # Fetch batch input file from bubble
    batch_input_file_name = batch_url.split('/')[-1]
    local_batch_path = f'{LAMBDA_DATA_DIR}/{batch_input_file_name}'

    try:
        get_bubble_doc(batch_url, local_batch_path)
        print("Retrieved batch doc from bubble")
    except:
        print("Failed to fetch batch doc from bubble")
        pass

    # load batch list
    batch_input_path = pathlib.Path(local_batch_path)
    with open(batch_input_path, encoding="utf-8") as f:
        batch_list = f.read()

    print('batch_list:')
    print(batch_list.split('\n'))

    for input_val in batch_list.split('\n'):
        print(input_val)
        # get workflow inputs
        input_vals = prep_input_vals([input_vars], [input_val], workflow)
        print('prepped input val:')
        print(input_vals)

        if 'sqs' in task_args:
            queue = SQS(task_args['sqs'])
            workflow.run_all(input_vals, bubble=False)
            queue.send({
                'output': workflow.steps[-1].output,
                'input_word_cnt': workflow.input_word_cnt,
                'output_word_cnt': workflow.output_word_cnt
            })
        else:
            workflow.run_all(input_vals, bubble=False)

            # send result to Bubble Document
            body = {
                'name': workflow.steps[-1].output[:25],
                'text': workflow.steps[-1].output,
                'user_email': email,
                'project': project_id
            }
            res = create_bubble_object('document', body)
            new_doc_id = res.json()['id']

            # add new doc to project
            res = get_bubble_object('project', project_id)
            project_docs = res.json()['response']['documents']

            _ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]})

            # Send usage to Bubble Workflow Runs
            body = {
                'workflow': workflow_id,
                'input_word_cnt': workflow.input_word_cnt,
                'output_word_cnt': workflow.output_word_cnt,
                'user_email': email
            }
            _ = create_bubble_object('workflow-runs', body)
    
    # Delete batch input document
    _ = delete_bubble_object('batch-doc', batch_doc_id)




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_args', default=None, type=str)
  args, _ = parser.parse_known_args()
  print(args.task_args)

  task_args = json.loads(args.task_args)
    
  main(task_args)

  