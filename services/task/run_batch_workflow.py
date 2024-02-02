import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import argparse
import pathlib
import boto3

from utils.workflow import get_workflow_from_bubble
from utils.bubble import create_bubble_object, get_bubble_object, update_bubble_object, get_bubble_doc, delete_bubble_object
from utils.google import get_csv_lines
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
    
    input_var = task_args['input_vars']

    doc_id = task_args['doc_id']
    batch_url = task_args['batch_input_url']
    batch_doc_id = task_args['batch_doc_id']

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
    if local_batch_path.endswith('.csv'):
        batch_list = get_csv_lines(content=None, path=local_batch_path, delimiter=',', return_as_json=True)
        batch_list = [item for item in batch_list if item != '']
    else:
        batch_input_path = pathlib.Path(local_batch_path)
        with open(batch_input_path, encoding="utf-8") as f:
            batch_list = f.read()

        if "<SPLIT>" in batch_list:
            splitter = "<SPLIT>"
        else:
            splitter = "\n"

        batch_list = batch_list.split(splitter)
        batch_list = [item for item in batch_list if item != '']

    
    for input_val in batch_list:
        workflow = get_workflow_from_bubble(workflow_id, email=email, doc_id=doc_id)
        print(f"batch item input: {input_val}")

        if 'sqs' in task_args:
            queue = SQS(task_args['sqs'])
            workflow.run_all([input_var], [input_val], bubble=False)
            queue.send({
                'order': 0,
                'output': workflow.steps[-1].output,
                'input_word_cnt': workflow.input_word_cnt,
                'output_word_cnt': workflow.output_word_cnt
            })
        else:
            workflow.run_all([input_var], [input_val], bubble=False)

            if task_args['to_project']:
                # send result to Bubble Document
                body = {
                    'name': input_val.replace('\n','_').replace(' ','_')[:50],
                    'text': workflow.steps[-1].output,
                    'user_email': email,
                    'project': project_id
                }
                res = create_bubble_object('document', body)
                new_doc_id = res.json()['id']

                # add new doc to project
                res = get_bubble_object('project', project_id)
                try:
                    project_docs = res.json()['response']['documents']
                except:
                    project_docs = []

                _ = update_bubble_object('project', project_id, {'documents': project_docs+[new_doc_id]})

            # Send usage to Bubble Workflow Runs
            body = {
                'workflow': workflow_id,
                'input_word_cnt': workflow.input_word_cnt,
                'output_word_cnt': workflow.output_word_cnt,
                'user_email': email
            }
            _ = create_bubble_object('workflow-runs', body)
    
    # Delete batch input document if present
    try:
        _ = delete_bubble_object('batch-doc', batch_doc_id)
    except:
        pass




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--task_args', default=None, type=str)
  args, _ = parser.parse_known_args()
  print(args.task_args)

  task_args = json.loads(args.task_args)
    
  main(task_args)

  