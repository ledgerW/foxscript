import sys
sys.path.append('..')

import os
from dotenv import load_dotenv
load_dotenv()

import json
import time
import re
import boto3
import requests
import weaviate as wv
from datetime import datetime
#from youtubesearchpython import VideosSearch

from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever


from pydantic import BaseModel, Field, create_model
from typing import List

from utils.workflow import get_top_n_search, cloud_research, get_wv_class_name
from utils.general import SQS
from utils.response_lib import *


import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")

if os.getenv('IS_OFFLINE'):
   boto3.setup_default_session(profile_name='ledger')
   lambda_client = boto3.client('lambda', endpoint_url=os.getenv('LOCAL_INVOKE_ENDPOINT'))
else:
   lambda_client = boto3.client('lambda')

STAGE = os.getenv('STAGE')
BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')
BUBBLE_API_ROOT = os.getenv('BUBBLE_API_ROOT')

WP_API_KEY = os.getenv('WP_API_KEY')
BUCKET = os.getenv('BUCKET')
LAMBDA_DATA_DIR = '/tmp'


auth_config = wv.auth.AuthApiKey(api_key=os.environ['WEAVIATE_API_KEY'])

wv_client = wv.Client(
    url=os.environ['WEAVIATE_URL'],
    additional_headers={
        "X-OpenAI-Api-Key": os.environ['OPENAI_API_KEY']
    },
    auth_client_secret=auth_config
)


# ACTIONS
class get_chain():
    def __init__(self, prompt=None):
        llm = ChatOpenAI(model_name="gpt-4", temperature=1.0)

        self.input_vars = re.findall('{(.+?)}', prompt)

        _prompt = PromptTemplate(
            input_variables=self.input_vars,
            template=prompt
        )

        self.chain = LLMChain(llm=llm, prompt=_prompt, verbose=True)

    def __call__(self, input):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        res = self.chain(input)

        return res['text']
    

class extract_from_text():
  def create_chain_parser_class(self, attributes):
    fields = {"__base__": BaseModel}

    for name, description in attributes.items():
        fields[name] = (List[str], Field(default=[], description=description))

    return create_model("ChainParser", **fields)
  
  
  def __init__(self, attributes=None):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1.0)

    self.attributes = attributes
    ChainParser = self.create_chain_parser_class(attributes)

    self.parser = PydanticOutputParser(pydantic_object=ChainParser)

    prompt = PromptTemplate(
      template="{format_instructions}\n\nGiven the below text, extract the information described in the output schema.\n{input}",
      input_variables=["input"],
      partial_variables={"format_instructions": self.parser.get_format_instructions()},
    )

    self.chain = LLMChain(llm=llm, prompt=prompt, verbose=True)


  def __call__(self, input):
    """
    Input: {'input': "input"}

    Returns: List[string]
    """
    input['input'] = "Original Text:\n" + input['input']
    res = self.chain(input)
    parsed_output = self.parser.parse(res['text']).dict()
    
    return parsed_output[list(self.attributes.keys())[0]]
  

#class get_yt_url():
#  def __init__(self, n=1):
#    self.n = n
#    self.input_vars = ['query']
  
#  def __call__(self, input):
#    """
#    Input: {'input': "query"}

#    Returns: string
#    """
#    query = input['input']

#    vid_search = VideosSearch(query, limit=self.n)

#    return vid_search.result()['result'][0]['link']
  

class do_research():
  def __init__(self, top_n=3):
    self.top_n = top_n
  

  def __call__(self, input):
    """
    Input: {'input': ["questions"]}

    Returns string
    """
    questions = input['input']

    urls_to_scrape = []
    queries = []
    for _query in questions:
      query = _query.strip()
      print(query)

      # top n search context
      try:
        top_n_search_results = get_top_n_search(query, self.top_n)
      except:
        try:
          time.sleep(3)
          top_n_search_results = get_top_n_search(query, self.top_n)
        except:
          continue

      # Collect each url for this query
      for _url in top_n_search_results:
        url = _url['link']
        urls_to_scrape.append(url)
        queries.append(query)

    # Scrape and Research all URLs concurrently
    sqs = 'research{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
    queue = SQS(sqs)
    for url, query in zip(urls_to_scrape, queries):
      cloud_research(url, sqs, query)
      time.sleep(3)
    
    # wait for and collect scrape results from SQS
    research_context = queue.collect(len(urls_to_scrape), max_wait=600)
    return '\n'.join(research_context)
  

class get_library_retriever():
  def __init__(self, class_name=None, k=3):
    self.retriever = self.get_weaviate_retriever(class_name=class_name, k=k)


  def get_weaviate_retriever(self, class_name=None, k=3):
    retriever = WeaviateHybridSearchRetriever(
        client=wv_client,
        index_name=f"{class_name}Chunk",
        text_key="chunk",
        k=k,
        attributes=['page', "fromContent {{... on {}Content {{ source url }}}}".format(class_name)],
        create_schema_if_missing=False
      )
    
    return retriever


  def get_library_chunks(self, query):
    chunks = self.retriever.get_relevant_documents(query)

    return chunks
  
  
  def __call__(self, input):
    """
    Input: {'input': ["questions"]}

    Returns: string
    """
    questions = input['input']
    
    all_results = ''
    for question in questions:
        all_results = all_results + question + '\n'
        chunks = self.get_library_chunks(question)
        results = '\n'.join([c.page_content for c in chunks])
        all_results = all_results + results + '\n\n'

    return all_results
  

class get_workflow():
    def __init__(self, workflow=None, config=None):
        if workflow:
            self.workflow = workflow

        if config:
            self.workflow = Workflow().load_from_config(config)

    def __call__(self, inputs):
        """
        Input: {
            'input_var_name': "input_var_val",
            'input_var2_name': "input_var2_val",
            ...
        }

        Returns: string
        """
        input_key = list(inputs.keys())[0]
        input_vals = list(inputs.values())[0]
        if type(input_vals) == list:
            sqs = 'workflow{}'.format(datetime.now().isoformat().replace(':','_').replace('.','_'))
            queue = SQS(sqs)
            for input in input_vals:
                payload = {
                    "body": {
                    'workflow_id': self.workflow.bubble_id,
                    'email': self.workflow.email,
                    'doc_id': '',
                    'input_vars': input_key,
                    'input_vals': input,
                    'sqs': sqs
                    }
                }

                if os.getenv('IS_OFFLINE'):
                    print('\nWorkflow payload would be:')
                    print(payload)
                    print('\n')
                else:
                    _ = lambda_client.invoke(
                        FunctionName=f'foxscript-api-{STAGE}-workflow',
                        InvocationType='Event',
                        Payload=json.dumps(payload)
                    )
                    time.sleep(5)
          
            if os.getenv('IS_OFFLINE'):
                return 'Dummy OFFLINE Output'
            else:
                workflow_outputs = queue.collect(len(input_vals), max_wait=600)
                return '\n\n'.join(workflow_outputs)
        else:
            self.workflow.run_all(inputs)
            return self.workflow.steps[-1].output
  

# Step Actions.
# A Step Action is a function that returns (func, [inputs_names])
ACTIONS = {
    'LLM Prompt': {
        'func': get_chain,
        'returns': 'string'
    },
    'Extract From Text': {
        'func': extract_from_text,
        'returns': 'list'
    },
    'Web Research': {
        'func': do_research,
        'returns': 'string'
    },
    'Library Research': {
        'func': get_library_retriever,
        'returns': 'string'
    },
    'Workflow': {
        'func': get_workflow,
        'returns': 'string'
    }
}


# WORKFLOW
class Workflow():
    def __init__(self, name=None):
        self.name = name
        self.steps = []
        self.output = {}
        self.user_inputs = {}

    def __repr__(self):
        step_repr = ["  Step {}. {}".format(i+1, s.name) for i, s in enumerate(self.steps)]
        return "\n".join([f'{self.name} Workflow'] + step_repr)

    def add_step(self, step):
        self.steps.append(step)

    def load_from_config(self, config):
        self.__init__(config['name'])

        try:
            self.bubble_id = config['workflow_id']
            self.email = config['email']
        except:
            pass
        
        for step_config in config['steps']:
            self.add_step(Step(step_config))

        return self

    def dump_config(self):
        return {
            "name": self.name,
            "steps": [s.config for s in self.steps]
        }
    
    def get_input_from_source(self, input_source, input_type):
        if "User Input" in input_source:
            input = self.user_inputs[input_source]

            if input_type == "Web Research":
                input = input[0]

            if input_type == "Library Research":
                input = input[0]

            return input
        else:
            step = [s for s in self.steps if s.name == input_source][0]
            return step.output

            
    def run_step(self, step_number, user_inputs={}):
        """
        input (str or list(str)): the input to the step (not in dictionary form) 
        """
        self.user_inputs = user_inputs
        
        # Get Step
        step = self.steps[step_number-1]
        print('{} - {} - {}'.format(step_number, step.config['step'], step.name))

        step_input = {
            input_var: self.get_input_from_source(input_source, step.config['action']) for input_var, input_source in step.config['inputs'].items()
        }

        step.run_step(step_input)


    def run_all(self, user_inputs, bubble=False):
        """
        user_inputs (dict)
        """
        self.user_inputs = user_inputs

        print('user_inputs')
        print(self.user_inputs)
        
        for step in self.steps:
            print('{} - {}'.format(step.config['step'], step.name))

            if step.config['action'] == 'Workflow':
               print('doing Workflow Step')
               input_var, input_source = list(step.config['inputs'].items())[0]
               step_workflow_input_var = list(step.func.workflow.steps[0].config['inputs'].values())[0]
               step_workflow_input_val = self.get_input_from_source(input_source, step.config['action'])
               step_input = prep_input_vals([step_workflow_input_var], [step_workflow_input_val], step.func.workflow)
            else:
                print('doing Normal Step')
                step_input = {
                    input_var: self.get_input_from_source(input_source, step.config['action']) for input_var, input_source in step.config['inputs'].items()
                }
                print('input_var and source: {}'.format(step.config['inputs'].items()))
                print('step_input: {}'.format(step_input))

            step.run_step(step_input)
            print(step.output)

            # Write each step output back to Bubble
            if bubble:
                if type(step.output) == list:
                   output = '\n'.join(step.output)
                else:
                   output = step.output

                bubble_body = {}
                table = 'step'
                bubble_id = step.bubble_id
                bubble_body['output'] = output
                res = update_bubble_object(table, bubble_id, bubble_body)
               

    def parse(self):
        pass


class Step():
    def __init__(self, config):
        self.name = config['name']
        self.config = config
        self.func = ACTIONS[config['action']]['func'](**config['init'])
        self.output_type = config['output_type']
        self.bubble_id = config['bubble_id']
        self.output = None

    def __repr__(self):
        return f'Step - {self.name}'

    def run_step(self, inputs):
        self.output = self.func(inputs)
    

def write_to_bubble(table, body):
    endpoint = BUBBLE_API_ROOT + f'/{table}'

    res = requests.post(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res


def update_bubble_object(table, uid, body):
    endpoint = BUBBLE_API_ROOT + f'/{table}' + f'/{uid}'

    res = requests.patch(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'},
        json=body
    )

    return res


def get_init(body, email):
    if body['type'] == 'LLM Prompt':
        init = {'prompt': body['init_text']}

    if body['type'] == 'Web Research':
        init = {'top_n': int(body['init_number'])}

    if body['type'] == 'Library Research':
        class_name, account = get_wv_class_name(email, body['init_text'])
        init = {
            'class_name': class_name,
            'k': int(body['init_number'])
        }

    if body['type'] == 'Extract From Text':
        init = {
            'attributes': {
                'extraction': body['init_text']
            }
        }

    if body['type'] == 'Workflow':
        init = {'workflow': get_workflow_from_bubble(body['init_text'], email=email)}
        
    if body['type'] == 'Get YouTube URL':
        init = {'n': body['init_number']}

    return init


def prep_input_vals(input_vars, input_vals, input):
    # prep for a Workflow
    if hasattr(input, 'steps'):
        input_type = input.steps[0].config['action']

        if input_type == 'LLM Prompt':
            input_vals = {var: source for var, source in zip(input_vars, input_vals)}  

        if input_type == 'Web Research':
            input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
        
        if input_type == 'Library Research':
            input_vals = {input_vars[0]: [x.split('\n') for x in input_vals]}
        
        if input_type == 'Extract From Text':
            input_vals = {input_vars[0]: input_vals[0]}

        if input_type == 'Workflow':
            input_vals = {input_vars[0]: input_vals[0]}
    # prep for Step
    else:
        input_type = input.config['action']
        if input_type == 'LLM Prompt':
            input_vals = {var: source for var, source in zip(input_vars, input_vals)}  

        if input_type == 'Web Research':
            input_vals = {'input': [x.split('\n') for x in input_vals]}
        
        if input_type == 'Library Research':
            input_vals = {'input': [x.split('\n') for x in input_vals]}
        
        if input_type == 'Extract From Text':
            input_vals = {'input': input_vals[0]}

    return input_vals  


def step_config_from_bubble(bubble_step, email):
    step_config = {
        "name": bubble_step['name'],
        "step": bubble_step['step_number'],
        "action": bubble_step['type'],
        "init": get_init(bubble_step, email),
        "inputs": {var: src for var, src in zip(bubble_step['input_vars'], bubble_step['input_vars_sources']) if var},
        "bubble_id": bubble_step['_id'],
        "output_type": "string" if bubble_step['type'] != 'Extract From Text' else 'list'
    }

    return step_config


def get_step_from_bubble(step_id, email=None):
    endpoint = BUBBLE_API_ROOT + '/step' + f'/{step_id}'
    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'}
    )
    bubble_step = json.loads(res.content)['response']

    return Step(step_config_from_bubble(bubble_step, email))


def get_workflow_from_bubble(workflow_id, email=None):
    # get workflow data from bubble db
    endpoint = BUBBLE_API_ROOT + '/workflow' + f'/{workflow_id}'
    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'}
    )
    bubble_workflow = json.loads(res.content)['response']

    # get workflow steps data from bubble db
    constraints = json.dumps([{"key":"workflow", "constraint_type": "equals", "value": workflow_id}])
    endpoint = BUBBLE_API_ROOT + '/step' + '?constraints={}'.format(constraints) + '&sort_field=step_number'
    res = requests.get(
        endpoint,
        headers={'Authorization': f'Bearer {BUBBLE_API_KEY}'}
    )
    bubble_steps = json.loads(res.content)['response']['results']
    step_configs = [step_config_from_bubble(step, email) for step in bubble_steps]

    workflow_config = {
        'name': bubble_workflow['name'],
        'steps': step_configs,
        'workflow_id': workflow_id,
        'email': email
    }

    return Workflow().load_from_config(workflow_config)


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
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]

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
        step.run_step(inputs)
        output = step.output

    # output prep
    if type(output) == list:
        output = '\n'.join(output)

    body = {
        'output': output
    }

    _ = update_bubble_object('step', step.bubble_id, body)
       
    return success({'SUCCESS': True})