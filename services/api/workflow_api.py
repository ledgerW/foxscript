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
from youtubesearchpython import VideosSearch

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

lambda_client = boto3.client('lambda')

STAGE = os.getenv('STAGE')
BUBBLE_API_KEY = os.getenv('BUBBLE_API_KEY')

# !!!!!!!!!
BUBBLE_API_ROOT = os.getenv('BUBBLE_API_ROOT')
BUBBLE_API_ROOT = "https://foxscript.bubbleapps.io/version-test/api/1.1/obj"
# !!!!!!!!!

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


llm = ChatOpenAI(model_name="gpt-4", temperature=0.8)



# ACTIONS
class get_chain():
    def __init__(self, prompt=None):
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
    res = self.chain(input)
    parsed_output = self.parser.parse(res['text']).dict()
    
    return parsed_output[list(self.attributes.keys())[0]]
  

class get_yt_url():
  def __init__(self, n=1):
    self.n = n
    self.input_vars = ['query']
  
  def __call__(self, input):
    """
    Input: {'input': "query"}

    Returns: string
    """
    query = input['input']

    vid_search = VideosSearch(query, limit=self.n)

    return vid_search.result()['result'][0]['link']
  

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
  

# Step Actions.
# A Step Action is a function that returns (func, [inputs_names])
ACTIONS = {
    'LLM Prompt': {
        'func': get_chain,
        'returns': 'string'
    },
    'Get YouTube URL': {
        'func': get_yt_url,
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
    }
}


# WORKFLOW
class Workflow():
    def __init__(self, name=None):
        self.name = name
        self.steps = []
        self.output = {}

    def __repr__(self):
        step_repr = ["  Step {}. {}".format(i+1, s.name) for i, s in enumerate(self.steps)]
        return "\n".join([f'{self.name} Workflow'] + step_repr)

    def add_step(self, step):
        self.steps.append(step)

    def load_from_config(self, config):
        self.__init__(config['name'])
        
        for step_config in config['steps']:
            self.add_step(Step(step_config))

        return self

    def dump_config(self):
        return {
            "name": self.name,
            "steps": [s.config for s in self.steps]
        }
            
    def run_step(self, step_number, input=None):
        if input and step_number==1:
            self.output[0] = input

        step = self.steps[step_number-1]
        print('{} - {} - {}'.format(step_number, step.config['step'], step.name))

        # collect input from previous step outputs according to this step config
        if step_number==1 and type(self.output[0]) == list:
           step_input = {k: v for k, v in zip(step.config['inputs'].keys(), self.output[0])}
        else:
           step_input = {k: self.output[step_number] for k, step_number in step.config['inputs'].items()}

        # execute this step and save to workflow output
        self.output[step.config['step']] = step.run_step(step_input)


    def run_all(self, input, bubble_body=None):
        self.output[0] = input
        for step_number in range(1, len(self.steps)+1):
            self.run_step(step_number)
            step = self.steps[step_number-1]

            if bubble_body:
                table = 'step'
                bubble_id = step.bubble_id
                bubble_body['output'] = self.output[step_number]
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

    def __repr__(self):
        return f'Step - {self.name}'

    def run_step(self, inputs):
        return self.func(inputs)
    

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

    if body['type'] == 'Get YouTube URL':
        init = {'n': body['init_number']}

    return init


def prep_input_vals(input_vals, input_type):
    if input_type == 'Web Research':
        input_vals = [x.split('\n') for x in input_vals]
    
    if input_type == 'Library Research':
        input_vals = [x.split('\n') for x in input_vals]

    return input_vals


def get_step_config(body, email):
    config = {}
    config['bubble_id'] = body['bubble_id']
    config['name'] = body['name']
    config['step'] = int(body['step_number'])
    config['action'] = body['type']
    config['init'] = get_init(body, email)
    config['output_type'] = body['output_type']

    return config


def get_step_inputs(body, steps):
    body['input_vars'] = [x.strip() for x in body['input_vars'].split(',') if x]
    body['input_vars_sources'] = [x.strip() for x in body['input_vars_sources'].split(',') if x]

    inputs = {}
    for input_var, input_var_source_name in zip(body['input_vars'], body['input_vars_sources']):
        for step in steps:
            if 'User Input' in input_var_source_name:
               input_var_source_idx = 0
               break

            if step['name'] == input_var_source_name:
                input_var_source_idx = int(step['step_number'])
                break
            
        inputs[input_var] = input_var_source_idx

    return inputs
    
    


# Lambda Handler
def workflow(event, context):
    print(event)

    try:
        email = event['body']['email']
        workflow = event['body']['workflow']
        project = event['body']['project']
        doc_id = event['body']['doc_id']

        input_vals = event['body']['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]

        body = event['body']
        body['steps'] = [json.loads(step) for step in body['steps'].split('<SPLIT>')]
        input_vars = [x.strip() for x in body['input_vars'].split(',') if x]
    except:
        email = json.loads(event['body'])['email']
        workflow = json.loads(event['body'])['workflow']
        project = json.loads(event['body'])['project']
        doc_id = json.loads(event['body'])['doc_id']
        
        input_vals = json.loads(event['body'])['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]
        
        body = json.loads(event['body'])
        body['steps'] = [json.loads(step) for step in body['steps'].split('<SPLIT>')]
        input_vars = [x.strip() for x in body['input_vars'].split(',') if x]

    print(body)

    # build config
    config = {}
    config['name'] = body['workflow']
    config['steps'] = []
    for step_body in body['steps']:
        step_config = get_step_config(step_body, email)
        step_config['inputs'] = get_step_inputs(step_body, body['steps'])
        config['steps'].append(step_config)

    # execute this step and save to workflow output
    input_vals = prep_input_vals(input_vals, config['steps'][0]['action'])
    inputs = {k: v for k, v in zip(input_vars, input_vals)}
    
    # run step as lambda Event so we can return immediately and free frontend
    _ = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-run_workflow',
        InvocationType='Event',
        Payload=json.dumps({"body": {
            'doc_id': doc_id,
            'email': email,
            'workflow': workflow,
            'project': project,
            'config': config,
            'inputs': inputs
        }})
    ) 

    return success({'SUCCESS': True})


def run_workflow(event, context):
    print(event)

    try:
        doc_id = event['body']['doc_id']

        inputs = event['body']['inputs']
        config = event['body']['config']
    except:
        doc_id = json.loads(event['body'])['doc_id']
        
        inputs = json.loads(event['body'])['inputs']
        config = json.loads(event['body'])['config']
   
   
   # load and run workflow
    workflow = Workflow().load_from_config(config)

    # write individual step results to bubble as they complete
    workflow.run_all(inputs, bubble_body={})

    # send result to Bubble frontend db
    table = 'document'
    uid = doc_id
    body = {
        'name': workflow.output[len(workflow.steps)][:25],
        'text': workflow.output[len(workflow.steps)]
    }

    _ = update_bubble_object(table, uid, body)
       
    return success({'SUCCESS': True})


# Lambda Handler
def step(event, context):
    print(event)

    try:
        email = event['body']['email']
        workflow = event['body']['workflow']

        input_vals = event['body']['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]

        body = event['body']
        body['input_vars'] = [x.strip() for x in body['input_vars'].split(',') if x]
    except:
        email = json.loads(event['body'])['email']
        workflow = json.loads(event['body'])['workflow']
        
        input_vals = json.loads(event['body'])['input_vals']
        input_vals = [x.strip() for x in input_vals.split('<SPLIT>') if x]
        
        body = json.loads(event['body'])
        body['input_vars'] = [x.strip() for x in body['input_vars'].split(',') if x]

    print(body)

    # build config
    config = get_step_config(body, email)

    # execute this step and save to workflow output
    input_vals = prep_input_vals(input_vals, body['type'])
    inputs = {k: v for k, v in zip(body['input_vars'], input_vals)}
    
    # run step as lambda Event so we can return immediately and free frontend
    _ = lambda_client.invoke(
        FunctionName=f'foxscript-api-{STAGE}-run_step',
        InvocationType='Event',
        Payload=json.dumps({"body": {
            'email': email,
            'workflow': workflow,
            'name': body['name'],
            'inputs': inputs,
            'config': config
        }})
    ) 
       
    return success({'SUCCES': True})


def run_step(event, context):
    print(event)

    try:
        email = event['body']['email']
        workflow = event['body']['workflow']
        name = event['body']['name']

        inputs = event['body']['inputs']
        config = event['body']['config']
    except:
        email = json.loads(event['body'])['email']
        workflow = json.loads(event['body'])['workflow']
        name = json.loads(event['body'])['name']
        
        inputs = json.loads(event['body'])['inputs']
        config = json.loads(event['body'])['config']

    # get step
    step = Step(config)

    # execute this step and save to workflow output
    output = step.run_step(inputs)

    # output prep
    if type(output) == list:
       output = '\n'.join(output)


    # send result to Bubble frontend db
    table = 'step'
    bubble_id = step.bubble_id
    body = {
       'output': output
    }

    _ = update_bubble_object(table, bubble_id, body)
       
    return success({'SUCCESS': True})