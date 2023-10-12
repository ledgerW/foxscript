import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import json
import boto3
#from youtubesearchpython import VideosSearch

from utils.workflow import prep_input_vals, get_workflow_from_bubble, get_step_from_bubble

from utils.bubble import update_bubble_object
from utils.general import SQS
from utils.response_lib import *

import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

STAGE = os.getenv('STAGE')
BUCKET = os.getenv('BUCKET')

if os.getenv('IS_OFFLINE'):
   boto3.setup_default_session(profile_name='ledger')
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
   LAMBDA_DATA_DIR = '.'
else:
   lambda_client = boto3.client('lambda')
   LAMBDA_DATA_DIR = '/tmp'



# Lambda Handler
def workflow(event, context):
    print(event)

    try:
        workflow_id = event['body']['workflow_id']
        email = event['body']['email']
        doc_id = event['body']['doc_id']

        input_vars = event['body']['input_vars']
        input_vars = [x.strip() for x in input_vars.split(',') if x]

        input_vals = event['body']['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]

        body = event['body']
    except:
        workflow_id = json.loads(event['body'])['workflow_id']
        email = json.loads(event['body'])['email']
        doc_id = json.loads(event['body'])['doc_id']

        input_vars = json.loads(event['body'])['input_vars']
        input_vars = [x.strip() for x in input_vars.split(',') if x]
        
        input_vals = json.loads(event['body'])['input_vals']
        
        print(input_vals)
        try:
            input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]
        except:
            pass

        body = json.loads(event['body'])

    if 'sqs' in body:
       # this is a Workflow as Step running distributed
       out_body = {
            'workflow_id': workflow_id,
            'email': email,
            'doc_id': doc_id,
            'input_vars': input_vars,
            'input_vals': input_vals,
            'sqs': body['sqs']
        }
    else:
        out_body = {
            'workflow_id': workflow_id,
            'email': email,
            'doc_id': doc_id,
            'input_vars': input_vars,
            'input_vals': input_vals
        }
    
    # run step as lambda Event so we can return immediately and free frontend
    _ = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-run_workflow',
        InvocationType='Event',
        Payload=json.dumps({"body": out_body})
    ) 

    return success({'SUCCESS': True})


def run_workflow(event, context):
    print(event)

    try:
        workflow_id = event['body']['workflow_id']
        email = event['body']['email']
        doc_id = event['body']['doc_id']
        input_vars = event['body']['input_vars']
        input_vals = event['body']['input_vals']
        body = event['body']
    except:
        workflow_id = json.loads(event['body'])['workflow_id']
        email = json.loads(event['body'])['email']
        doc_id = json.loads(event['body'])['doc_id']
        input_vars = json.loads(event['body'])['input_vars']
        input_vals = json.loads(event['body'])['input_vals']
        body = json.loads(event['body'])
   
   
   # load and run workflow
    workflow = get_workflow_from_bubble(workflow_id, email=email)

    # get workflow inputs
    input_vals = prep_input_vals(input_vars, input_vals, workflow)

    if 'sqs' in body:
        # running as a distributed step, send output back to master
        # there is no doc_id because output returns to the calling step
        queue = SQS(body['sqs'])
        workflow.run_all(input_vals, bubble=False)
        queue.send(workflow.steps[-1].output)
    else:
        # write individual step results to bubble as they complete
        workflow.run_all(input_vals, bubble=True)

    if doc_id:
        # send result to Bubble frontend db
        table = 'document'
        uid = doc_id
        body = {
            'name': workflow.steps[-1].output[:25],
            'text': workflow.steps[-1].output
        }

        _ = update_bubble_object(table, uid, body)
       
    return success({'SUCCESS': True})


# Lambda Handler
def step(event, context):
    print(event)

    try:
        step_id = event['body']['step_id']
        email = event['body']['email']

        input_vars = event['body']['input_vars']
        input_vars = [x.strip() for x in input_vars.split(',') if x]

        input_vals = event['body']['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]
    except:
        step_id = json.loads(event['body'])['step_id']
        email = json.loads(event['body'])['email']

        input_vars = json.loads(event['body'])['input_vars']
        input_vars = [x.strip() for x in input_vars.split(',') if x]

        input_vals = json.loads(event['body'])['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]

    
    # run step as lambda Event so we can return immediately and free frontend
    _ = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-run_step',
        InvocationType='Event',
        Payload=json.dumps({"body": {
            'step_id': step_id,
            'email': email,
            'input_vars': input_vars,
            'input_vals': input_vals
        }})
    ) 
       
    return success({'SUCCES': True})


def run_step(event, context):
    print(event)

    try:
        step_id = event['body']['step_id']
        email = event['body']['email']
        input_vars = event['body']['input_vars']
        input_vals = event['body']['input_vals']
    except:
        step_id = json.loads(event['body'])['step_id']
        email = json.loads(event['body'])['email']
        input_vars = json.loads(event['body'])['input_vars']
        input_vals = json.loads(event['body'])['input_vals']

    # get step
    step = get_step_from_bubble(step_id, email=email)

    if step.config['action'] == 'Workflow':
        input_var = list(step.func.workflow.steps[0].config['inputs'].values())[0]
        inputs = prep_input_vals([input_var], input_vals, step.func.workflow)
        step.func.workflow.run_all(inputs, bubble=False)
        output = step.func.workflow.steps[-1].output
    else:
        inputs = prep_input_vals(input_vars, input_vals, step)

        print('\nInputs:')
        print(inputs)
        print('\n')

        step.run_step(inputs)
        output = step.output

        print('\nOutput:')
        print(output)
        print('\n')

    # output prep for bubble display (doesn't change internal workflow output)
    if type(output) == list:
        output = '\n'.join(output)

    body = {
        'output': output
    }

    _ = update_bubble_object('step', step.bubble_id, body)
       
    return success({'SUCCESS': True})